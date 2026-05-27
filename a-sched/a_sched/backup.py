import glob
import psutil
import json
import os
from datetime import datetime

from a_sched.task import TaskManager
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils

# 亲和信息备份文件名前缀，完整文件名形如 cpu_affinity_backup_20260525_143022.json
AFFINITY_BACKUP_FILE_PREFIX = "cpu_affinity_backup_"
AFFINITY_BACKUP_FILE_SUFFIX = ".json"


class AffinityBackup:
    """亲和信息备份恢复"""

    def __init__(self, task: TaskManager, domain: AffinityDomainManager):
        self._task = task
        self._domain = domain

    def backup_affinity(self) -> None:
        affinity_data = {
            "cpu_bind_data": self.build_cpu_bind_data(),
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
                irqs = []
                if npu._sq_irq is not None:
                    irqs.append((npu._sq_irq, npu.SQ_IRQ))
                for cq in npu._cq_irqs:
                    irqs.append((cq, npu.CQ_IRQ))
                if npu._trs_mbox_irq is not None:
                    irqs.append((npu._trs_mbox_irq, npu.trs_mbox_name))
                for irq_id, irq_name in irqs:
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
                entry = {"npu_id": npu_id}
                if npu._dev_sq_task is not None:
                    try:
                        entry["dev_sq_task"] = {
                            "pid": npu._dev_sq_task,
                            "name": npu.dev_sq_task_name,
                            "cpu_affinity": utils.get_process_cpus_by_pid(pid=npu._dev_sq_task),
                        }
                    except Exception as e:
                        print(f"npu[{npu_id}] {npu.dev_sq_task_name} get cpu affinity failed, {str(e)}")
                if npu._dev_sq_send_wq is not None:
                    try:
                        entry["dev_sq_send_wq"] = {
                            "wq_name": npu.dev_sq_send_wq_name,
                            "cpu_affinity": utils.get_npu_work_queue_cpus_by_name(wq_name=npu.dev_sq_send_wq_name),
                        }
                    except Exception as e:
                        print(f"npu[{npu_id}] {npu.dev_sq_send_wq_name} get cpu affinity failed, {str(e)}")
                if len(entry) > 1:
                    dev_sq_bind_data[npu_id] = entry
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

    def print_affinity(self) -> None:
        """打印亲和任务中进程/线程当前实际的亲和信息"""

        for group in self._task.groups.values():
            print(f"TaskGroup[{group.group_id}]: {f'name={group.name}' if group.name else ''}")
            for pid, process in group.process_tasks.items():
                priority = process._priority_names.get(process.priority, "UNKNOWN")
                cpus = utils.get_process_cpus_by_pid(pid=pid)
                print(
                    f"  - PROCESS[{pid}]: name={process.name}, priority={priority}, "
                    f"socket={self._domain.get_sockets_of_cpus(cpus)}, "
                    f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                    f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                    f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                )
            for tid, thread in group.thread_tasks.items():
                priority = thread._priority_names.get(thread.priority, "UNKNOWN")
                cpus = utils.get_thread_cpus_by_tid(tid=tid)
                print(
                    f"  - THREAD[{tid}]: name={thread.name}, priority={priority}, "
                    f"socket={self._domain.get_sockets_of_cpus(cpus)}, "
                    f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                    f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                    f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                )
            for npu_id, npu in group.npu_tasks.items():
                print(f"  - NPU[{npu_id}]: ")
                if npu._dev_sq_task is not None:
                    cpus = utils.get_process_cpus_by_pid(pid=npu._dev_sq_task)
                    print(
                        f"    - Process[{npu._dev_sq_task}]({npu.dev_sq_task_name}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._dev_sq_send_wq is not None:
                    cpus = utils.get_npu_work_queue_cpus_by_name(wq_name=npu.dev_sq_send_wq_name)
                    print(
                        f"    - Process[{npu._dev_sq_send_wq}]({npu.dev_sq_send_wq_name}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._acl_thread is not None:
                    cpus = utils.get_thread_cpus_by_tid(tid=npu._acl_thread)
                    print(
                        f"    - Thread[{npu._acl_thread}]({npu.ACL_THREAD}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._release_thread is not None:
                    cpus = utils.get_thread_cpus_by_tid(tid=npu._release_thread)
                    print(
                        f"    - Thread[{npu._release_thread}]({npu.RELEASE_THREAD}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._rt_recycle_thread is not None:
                    cpus = utils.get_thread_cpus_by_tid(tid=npu._rt_recycle_thread)
                    print(
                        f"    - Thread[{npu._rt_recycle_thread}]({npu.RT_RECYCLE_THREAD}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._sq_irq is not None:
                    cpus = utils.get_irq_cpus_by_irq_id(irq_id=npu._sq_irq)
                    print(
                        f"    - Irq[{npu._sq_irq}]({npu.SQ_IRQ}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                for cq in npu._cq_irqs:
                    cpus = utils.get_irq_cpus_by_irq_id(irq_id=cq)
                    print(
                        f"    - Irq[{cq}]({npu.CQ_IRQ}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
                if npu._trs_mbox_irq is not None:
                    cpus = utils.get_irq_cpus_by_irq_id(irq_id=npu._trs_mbox_irq)
                    print(
                        f"    - Irq[{npu._trs_mbox_irq}]({npu.trs_mbox_name}): "
                        f"numa=[{utils.compress_continuous(self._domain.get_numas_of_cpus(cpus))}], "
                        f"cluster=[{utils.compress_continuous(self._domain.get_clusters_of_cpus(cpus))}], "
                        f"cpu=[{utils.CPUMask().from_list(cpus)}]"
                    )
