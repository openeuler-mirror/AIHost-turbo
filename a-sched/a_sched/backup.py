import glob
import psutil
import json
import os
from datetime import datetime

from a_sched.cpuset import CpusetManager
from a_sched.task import NpuTaskA3, TaskManager
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils

# 亲和信息备份文件名前缀，完整文件名形如 cpu_affinity_backup_20260525_143022.json
AFFINITY_BACKUP_FILE_PREFIX = "cpu_affinity_backup_"
AFFINITY_BACKUP_FILE_SUFFIX = ".json"


class AffinityBackup:
    """亲和信息备份恢复"""

    def __init__(self, task: TaskManager, domain: AffinityDomainManager, cpuset: CpusetManager):
        self._task = task
        self._domain = domain
        self._cpuset = cpuset

    def backup_affinity(self) -> None:
        affinity_data = {
            "cpu_bind_data": self.build_cpu_bind_data(),
            "cpuset_data": self._cpuset.get_backup_data(),
            "background_bind_data": self.build_background_bind_data(),
            "irq_bind_data": self.build_irq_bind_data(),
            "irq_service_status": self.build_irq_service_status(),
            "dev_sq_bind_data": self.build_dev_sq_bind_data(),
        }

        # 写入文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{AFFINITY_BACKUP_FILE_PREFIX}{timestamp}{AFFINITY_BACKUP_FILE_SUFFIX}"
        try:
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(affinity_data, f, indent=4, ensure_ascii=False)
            os.chmod(backup_file, 0o444)
            print(f"save affinity to {backup_file} SUCCESS!")
        except Exception as e:
            print(f"save affinity to {backup_file} FAIL! {str(e)}")

    def build_cpu_bind_data(self) -> dict:
        cpu_bind_data = {}
        for group in self._task.groups.values():
            for pid, process_task in group.process_tasks.items():
                try:
                    process = psutil.Process(pid)
                    threads = []
                    for thread in process.threads():
                        try:
                            threads.append(
                                {
                                    "tid": thread.id,
                                    "thread_name": utils.get_thread_name_by_tid(tid=thread.id, pid=pid),
                                    "cpu_affinity": utils.get_thread_cpus_by_tid(tid=thread.id, pid=pid),
                                }
                            )
                        except Exception as e:
                            print(f"thread [{thread.id}] get cpu affinity failed, {str(e)}")

                    try:
                        cpu_bind_data[pid] = {
                            "pid": pid,
                            "process_name": process.name(),
                            "cpu_affinity": process.cpu_affinity(),
                            "threads": threads,
                        }
                    except Exception as e:
                        print(f"process [{pid}-{process_task.name}] get cpu affinity failed, {str(e)}")

                except Exception as e:
                    print(f"process [{pid}-{process_task.name}] not found, {str(e)}")
        return cpu_bind_data

    def build_irq_service_status(self) -> bool:
        _, _, ret = utils.execute_command(["systemctl", "is-active", "--quiet", "irqbalance"])
        return True if ret == 0 else False

    def build_irq_bind_data(self) -> dict:
        irq_bind_data = {}
        for group in self._task.groups.values():
            for npu in group.npu_tasks.values():
                if not isinstance(npu, NpuTaskA3):
                    continue
                for irq_id, irq_name in npu.irqs:
                    try:
                        irq_bind_data[irq_id] = {
                            "irq_id": irq_id,
                            "irq_name": irq_name,
                            "cpu_affinity": utils.get_irq_cpus_by_irq_id(irq_id=irq_id),
                        }
                    except Exception as e:
                        print(f"irq [{irq_id}-{irq_name}] get cpu affinity failed, {str(e)}")
        return irq_bind_data

    def build_background_bind_data(self) -> dict:
        backgroud_bind_data = {}
        for pid, name in self._task.background_processes.items():
            try:
                proc = psutil.Process(pid)
                backgroud_bind_data[pid] = {
                    "pid": pid,
                    "process_name": proc.name(),
                    "cpu_affinity": proc.cpu_affinity(),
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return backgroud_bind_data

    def build_dev_sq_bind_data(self) -> dict:
        dev_sq_bind_data = {}
        for group in self._task.groups.values():
            for npu_id, npu in group.npu_tasks.items():
                if not isinstance(npu, NpuTaskA3):
                    continue
                bind_data = npu.build_dev_bind_data()
                if bind_data:
                    dev_sq_bind_data[npu_id] = bind_data
        return dev_sq_bind_data

    def restore_affinity(self) -> None:
        # 1. 查找最早的备份文件
        backup_files = sorted(glob.glob(f"{AFFINITY_BACKUP_FILE_PREFIX}*{AFFINITY_BACKUP_FILE_SUFFIX}"))
        if not backup_files:
            print(f"no backup file matching {AFFINITY_BACKUP_FILE_PREFIX}*{AFFINITY_BACKUP_FILE_SUFFIX} found")
            return
        backup_file = backup_files[0]
        print(f"restore from earliest backup file: {backup_file}")

        # 2. 读取保存数据
        try:
            with open(backup_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            if not saved_data:
                print("data is empty")
                return
        except json.JSONDecodeError:
            print(f"json decode error: {backup_file}")
            return

        # 3. 批量恢复
        self._cpuset.restore_cpuset(saved_data.get("cpuset_data", {}))
        self.restore_cpu_bind_data(saved_data.get("cpu_bind_data", {}))
        self.restore_irq_bind_data(saved_data.get("irq_bind_data", {}))
        self.restore_irq_service_status(saved_data.get("irq_service_status", False))
        self.restore_dev_sq_bind_data(saved_data.get("dev_sq_bind_data", {}))
        self.restore_background_bind_data(saved_data.get("background_bind_data", {}))

    def restore_background_bind_data(self, background_bind_data: dict) -> None:
        restored = 0
        for pid_str, data in background_bind_data.items():
            try:
                pid = int(pid_str)
                proc = psutil.Process(pid)
                orig_affinity = data["cpu_affinity"]
                proc.cpu_affinity(orig_affinity)
                for thread in proc.threads():
                    try:
                        psutil.Process(thread.id).cpu_affinity(orig_affinity)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                restored += 1
            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                print(f"background process [{pid_str}] restore failed, {str(e)}")

        if restored:
            print(f"restore {restored} background processes affinity SUCCESS!")

    def restore_cpu_bind_data(self, cpu_bind_data: dict) -> None:
        for pid, data in cpu_bind_data.items():
            try:
                pid = int(pid)
                proc = psutil.Process(pid)
                proc_name = data["process_name"]
                proc.cpu_affinity(data["cpu_affinity"])
                cpus = utils.CPUMask().from_list(proc.cpu_affinity())
                print(f"restore process-{pid}-{data['process_name']}, cpu_affinity:{cpus}")
                for thread_data in data["threads"]:
                    try:
                        thread = psutil.Process(thread_data["tid"])
                        thread.cpu_affinity(thread_data["cpu_affinity"])
                        cpus = utils.CPUMask().from_list(thread.cpu_affinity())
                        print(
                            f"restore thread-{thread_data['tid']}-{thread_data['thread_name']}(process-{proc_name}), cpu_affinity:{cpus}"
                        )
                    except Exception as e:
                        print(f"thread-{thread_data['thread_name']}(process-{proc_name}) restore failed, {str(e)}")

            except Exception as e:
                print(f"process [{pid}] restore failed, {str(e)}")

    def restore_irq_service_status(self, should_start: bool) -> None:
        if should_start:
            utils.execute_command(["systemctl", "start", "irqbalance"])
            print("restore irqbalance service to active.")

    def restore_irq_bind_data(self, irq_bind_data: dict) -> None:
        for irq_id, data in irq_bind_data.items():
            try:
                irq_id = int(irq_id)
                irq_name = data["irq_name"]
                cpus = data["cpu_affinity"]
                utils.bind_irq_to_cpus(irq_id=irq_id, cpus=cpus, irq_name=irq_name)
                print(f"restore irq-{irq_id}-{irq_name}, cpu_affinity: {utils.CPUMask().from_list(cpus)}")
            except Exception as e:
                print(f"irq [{irq_id}] restore failed, {str(e)}")

    def restore_dev_sq_bind_data(self, dev_sq_bind_data: dict) -> None:
        for npu_id, data in dev_sq_bind_data.items():
            task_info = data.get("dev_sq_task")
            if task_info:
                try:
                    pid = task_info["pid"]
                    cpus = task_info["cpu_affinity"]
                    utils.bind_process_to_cpus(pid=pid, cpus=cpus)
                    print(
                        f"restore npu[{npu_id}] {task_info['name']} - {pid}, "
                        f"cpu_affinity: {utils.CPUMask().from_list(cpus)}"
                    )
                except Exception as e:
                    print(f"npu[{npu_id}] {task_info.get('name')} restore failed, {str(e)}")

            wq_info = data.get("dev_sq_send_wq")
            if wq_info:
                try:
                    cpus = wq_info["cpu_affinity"]
                    utils.bind_npu_sq_send_wq_to_cpus(npu_id=int(npu_id), cpus=cpus)
                    print(f"restore npu[{npu_id}] {wq_info['wq_name']}, cpu_affinity: {utils.CPUMask().from_list(cpus)}")
                except Exception as e:
                    print(f"npu[{npu_id}] {wq_info.get('wq_name')} restore failed, {str(e)}")
