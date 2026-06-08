import os
from collections import defaultdict

from a_sched import utils
from a_sched.affinity_domain import AffinityDomainManager
from a_sched.task import TaskManager
from a_sched.config import AffinityConfig

CPUSET_V1_ROOT_PATH = "/sys/fs/cgroup/cpuset"
CPUSET_V2_ROOT_PATH = "/sys/fs/cgroup"


class CpusetManager:
    def __init__(self, config: AffinityConfig, task: TaskManager, domain: AffinityDomainManager):
        self._task = task
        self._domain = domain
        self._config = config

        if os.path.exists(os.path.join(CPUSET_V2_ROOT_PATH, "cgroup.controllers")):
            self.root_dir = CPUSET_V2_ROOT_PATH
            self._isolate_cpus_impl = self.isolate_cpus_v2
            self._get_backup_data_impl = self.get_backup_data_v2
            self._restore_cpuset_impl = self.restore_cpuset_v2
            self.cgroup_thread_file = "cgroup.threads"
            self.cgroup_process_file = "cgroup.procs"
        else:
            self.root_dir = CPUSET_V1_ROOT_PATH
            self._isolate_cpus_impl = self.isolate_cpus_v1
            self._get_backup_data_impl = self.get_backup_data_v1
            self._restore_cpuset_impl = self.restore_cpuset_v1
            self.cgroup_thread_file = "tasks"
            self.cgroup_process_file = "cgroup.procs"

        self.isolated_dir = os.path.join(self.root_dir, "isolated")
        self.shared_dir = os.path.join(self.root_dir, "shared")
        self.high_prio_threads = defaultdict(list)  # pid -> list(tid)
        self.normal_processes: list[int] = []

    def init_high_prio_threads(self) -> None:
        for group in self._task.groups.values():
            threads = group.get_high_prio_threads()
            for pid, tids in threads.items():
                self.high_prio_threads[pid].extend(tids)

    def init_normal_processes(self) -> None:
        for group in self._task.groups.values():
            normal_tasks = group.get_normal_prio_tasks()
            for task in normal_tasks:
                pid = utils.get_thread_pid_by_tid(task.task_id)
                if pid is not None:
                    self.normal_processes.append(pid)

    def isolate_cpus(self, isolated_cpus: list[int], shared_cpus: list[int]):
        if not self._config.enable_cpuset:
            return
        print(f"\nStarting isolate cpuset...")
        return self._isolate_cpus_impl(isolated_cpus, shared_cpus)

    def restore_cpuset(self, backup_data: dict) -> None:
        if not self._config.enable_cpuset:
            return
        print(f"\nStarting restore cpuset...")
        self._restore_cpuset_impl(backup_data)

    def get_backup_data(self) -> dict:
        if not self._config.enable_cpuset:
            return {}
        self.init_high_prio_threads()
        self.init_normal_processes()
        return self._get_backup_data_impl()

    def isolate_cpus_v1(self, isolated_cpus: list[int], shared_cpus: list[int]):
        # step1: 开启根组负载均衡
        if not utils.write_str_param(self.root_dir, "cpuset.sched_load_balance", "1"):
            return

        # step2: 创建隔离/共享子组
        if not utils.create_dir(self.root_dir, "isolated"):
            return
        if not utils.create_dir(self.root_dir, "shared"):
            return

        # step3: 设置隔离/共享子组cpu
        isolated_cpus_str = utils.compress_continuous(isolated_cpus)
        if not utils.write_str_param(self.isolated_dir, "cpuset.cpus", isolated_cpus_str):
            return
        shared_cpus_str = utils.compress_continuous(shared_cpus)
        if not utils.write_str_param(self.shared_dir, "cpuset.cpus", shared_cpus_str):
            return

        # step4: 设置隔离/共享组内存节点
        numa_list = self._domain.get_numas_of_cpus(isolated_cpus)
        cgroup_mems = utils.compress_continuous(numa_list)
        if not utils.write_str_param(self.isolated_dir, "cpuset.mems", cgroup_mems):
            return
        numa_list = self._domain.get_numas_of_cpus(shared_cpus)
        cgroup_mems = utils.compress_continuous(numa_list)
        if not utils.write_str_param(self.shared_dir, "cpuset.mems", cgroup_mems):
            return

        # step5: 从已存在的子组中过滤cpu
        self.filter_out_isolated_cpus_from_existed_groups(isolated_cpus + shared_cpus)

        # step6: 普通任务迁移至共享子组
        self.move_processes_to_shared_group(self.normal_processes)

        # step7: 重新绑定普通任务优先级
        for group in self._task.groups.values():
            for task in group.get_normal_prio_tasks():
                task.bind_cpu()

        # step8: 高优先级线程迁移至隔离组
        self.move_threads_to_isolated_group()

        # step9: 设置隔离子组为独占模式
        utils.write_str_param(self.isolated_dir, "cpuset.cpu_exclusive", "1")

        # step10: 关闭隔离子组负载均衡
        utils.write_str_param(self.isolated_dir, "cpuset.sched_load_balance", "0")

        # step11: 开启共享子组负载均衡
        utils.write_str_param(self.shared_dir, "cpuset.sched_load_balance", "1")

    def isolate_cpus_v2(self, isolated_cpus: list[int], shared_cpus: list[int]):
        # step1: 使能子组cpuset
        if not self.enable_cpuset_subtree_control():
            return

        # step2: 创建隔离子组
        if not utils.create_dir(self.root_dir, "isolated"):
            return

        # step3: 创建进程共享子组
        if not utils.create_dir(self.root_dir, "shared"):
            return

        # step4: 设置隔子组cpu
        isolated_cpus_str = utils.compress_continuous(isolated_cpus)
        if not utils.write_str_param(self.isolated_dir, "cpuset.cpus", isolated_cpus_str):
            return
        shared_cpus_str = utils.compress_continuous(shared_cpus)
        if not utils.write_str_param(self.shared_dir, "cpuset.cpus", shared_cpus_str):
            return

        # step5: 设置创建的子组为线程模式
        if utils.write_str_param(self.isolated_dir, "cgroup.type", "threaded"):
            print(f"success to set isolated cgroup threaded")
        if utils.write_str_param(self.shared_dir, "cgroup.type", "threaded"):
            print(f"success to set shared cgroup threaded")

        # step6: 高优先级线程的父进程&普通优先级线程迁移至共享组
        self.move_processes_to_shared_group(self.high_prio_threads.keys())
        self.move_processes_to_shared_group(self.normal_processes)

        # step7: 高优先级线程迁移至隔离组
        self.move_threads_to_isolated_group()

        # step8: 设置子组分区状态为isolated
        utils.write_str_param(self.isolated_dir, "cpuset.cpus.partition", "isolated")
        utils.write_str_param(self.shared_dir, "cpuset.cpus.partition", "isolated")

    def move_threads_to_isolated_group(self) -> None:
        task_path = os.path.join(self.isolated_dir, self.cgroup_thread_file)
        try:
            with open(task_path, "a") as f:
                for tids in self.high_prio_threads.values():
                    for tid in tids:
                        try:
                            f.write(f"{tid}\n")
                            f.flush()
                            print(f"success to move {tid} to {task_path}")
                        except Exception as e:
                            print(f"error: failed to move {tid} to {task_path} -> {e}")
        except Exception as e:
            print(f"error: failed to open {task_path} -> {e}")

    def move_processes_to_shared_group(self, pids) -> None:
        task_path = os.path.join(self.shared_dir, self.cgroup_process_file)
        try:
            with open(task_path, "a") as f:
                for pid in pids:
                    try:
                        f.write(f"{pid}\n")
                        f.flush()
                        print(f"success to move {pid} to {task_path}")
                    except Exception as e:
                        print(f"error: failed to move {pid} to {task_path} -> {e}")
        except Exception as e:
            print(f"error: failed to open {task_path} -> {e}")

    def filter_out_isolated_cpus_from_existed_groups(self, isolated_cpus: list[int]) -> None:
        for cgroup_name in utils.safe_listdir(self.root_dir):
            all_cgroup_dirs = []
            cgroup_dir = os.path.join(self.root_dir, cgroup_name)
            # 收集所有包含cpuset.cpus的目录
            for dir_path, _, _ in os.walk(cgroup_dir):
                if os.path.exists(os.path.join(dir_path, "cpuset.cpus")):
                    all_cgroup_dirs.append(dir_path)

            # 倒叙：子目录优先（路径越长越先执行）
            all_cgroup_dirs.sort(key=lambda x: x.count("/"), reverse=True)

            # 按 先子后父 顺序修改
            for cg_dir in all_cgroup_dirs:
                original_cps, ret = utils.read_list_param(cg_dir, "cpuset.cpus")
                if not ret:
                    continue
                # 只保留非隔离核
                new_cpus = [cpu for cpu in original_cps if cpu not in isolated_cpus]
                cpus_str = utils.compress_continuous(new_cpus)
                if utils.write_str_param(cg_dir, "cpuset.cpus", cpus_str):
                    print(f"success to set cpus({cpus_str}) to {cg_dir}/cpuset.cpus")

    def enable_cpuset_subtree_control(self) -> bool:
        subtree_control = os.path.join(self.root_dir, "cpuset.subtree_control")

        if not os.path.exists(subtree_control):
            print(f"cpuset.subtree_control not exists")
            return False

        with open(subtree_control, "r", encoding="utf-8") as f:
            enabled = set(f.read().split())

        if "cpuset" in enabled:
            print(f"cpuset.subtree_control is already enabled")
            return True

        try:
            with open(subtree_control, "w", encoding="utf-8") as f:
                f.write("+cpuset\n")
        except Exception as e:
            print(f"error: failed to add cpuset to {subtree_control} -> {e}")
            return False

        print(f"success to add cpuset to {subtree_control}")
        return True

    def get_backup_data_v1(self) -> dict:
        backup_data = {}
        root_load_balance, ret = utils.read_str_param(self.root_dir, "cpuset.sched_load_balance")
        if ret:
            backup_data["root_load_balance"] = root_load_balance
        backup_data["cgroup"] = self.get_cgroups_data()
        backup_data["threads_tasks"], backup_data["processes_tasks"] = self.get_tasks_data_v1()
        return backup_data

    def get_backup_data_v2(self) -> dict:
        return {
            "processes_tasks": self.get_tasks_data_v2(),
        }

    def get_cgroups_data(self):
        cgroups = {}
        for cgroup_name in utils.safe_listdir(self.root_dir):
            cgroup_dir = os.path.join(self.root_dir, cgroup_name)
            for dir_path, _, _ in os.walk(cgroup_dir):
                cpus_file = os.path.join(dir_path, "cpuset.cpus")
                if os.path.exists(cpus_file):
                    try:
                        with open(cpus_file, "r", encoding="utf-8") as f:
                            cpus = f.read().strip()
                            cgroups[dir_path] = cpus
                    except Exception as e:
                        print(f"error: failed to read cpus from {dir_path} -> {e}")
        return cgroups

    def get_tasks_data_v1(self) -> tuple[dict, dict]:
        threads_tasks_data = {}
        for pid, tids in self.high_prio_threads.items():
            self.record_tasks(pid, tids, threads_tasks_data)
        processes_tasks_data = {}
        for pid in self.normal_processes:
            self.record_tasks(pid, [pid], processes_tasks_data)
        return threads_tasks_data, processes_tasks_data

    def get_tasks_data_v2(self) -> dict:
        processes_tasks_data = {}
        for pid in self.high_prio_threads.keys():
            self.record_tasks(pid, [pid], processes_tasks_data)
        for pid in self.normal_processes:
            self.record_tasks(pid, [pid], processes_tasks_data)
        return processes_tasks_data

    def record_tasks(self, pid: int, task_list: list[int], tasks_data: dict) -> None:
        for task_id in task_list:
            try:
                # 读取线程的cgroup信息
                cgroup_path = f"/proc/{pid}/task/{task_id}/cgroup"
                with open(cgroup_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # 提取cpuset子组的路径
                cpuset_relative_path = ""
                for line in lines:
                    if "cpuset" in line:
                        # v1格式如：11:cpuset/isolated
                        cpuset_relative_path = line.split(":")[2].strip().lstrip("/")
                        break
                    elif line.startswith("0::"):
                        # v2格式固定0::开头
                        cpuset_relative_path = line[3:].strip().lstrip("/")
                        break
                if not cpuset_relative_path:
                    print(f"error: failed to read cpuset relative path from {pid}")
                    continue
                tasks_data[task_id] = cpuset_relative_path
            except Exception as e:
                print(f"error: failed to read cpuset relative path from {pid} -> {e}")

    def restore_cpuset_v1(self, backup_data: dict) -> None:
        # 先关闭隔离组独占模式
        utils.write_str_param(self.isolated_dir, "cpuset.cpu_exclusive", "0")

        # 恢复根组负载均衡开关
        root_load_balance = backup_data.get("root_load_balance", None)
        if root_load_balance is not None:
            utils.write_str_param(self.root_dir, "cpuset.sched_load_balance", root_load_balance)

        # 恢复其他子组cpu
        cgroups = backup_data.get("cgroup", {})
        for cgroup_dir, cpus in cgroups.items():
            utils.write_str_param(cgroup_dir, "cpuset.cpus", cpus)

        # 恢复隔离线程至原组
        self.restore_cgroup_tasks(backup_data.get("processes_tasks", {}), self.cgroup_process_file)
        self.restore_cgroup_tasks(backup_data.get("threads_tasks", {}), self.cgroup_thread_file)

        # 删除隔离/共享组
        utils.remove_dir(self.root_dir, "isolated")
        utils.remove_dir(self.root_dir, "shared")

    def restore_cpuset_v2(self, backup_data: dict) -> None:
        # 关闭隔离组独占模式
        utils.write_str_param(self.isolated_dir, "cpuset.cpus.partition", "member")
        utils.write_str_param(self.shared_dir, "cpuset.cpus.partition", "member")

        # 恢复迁出进程及其子线程至原组
        self.restore_cgroup_tasks(backup_data.get("processes_tasks", {}), self.cgroup_process_file)

        # 删除隔离组、共享组
        utils.remove_dir(self.root_dir, "isolated")
        utils.remove_dir(self.root_dir, "shared")

    def restore_cgroup_tasks(self, tasks: dict, file_name: str) -> None:
        for task_id, cgroup_name in tasks.items():
            cgroup_path = os.path.join(self.root_dir, cgroup_name)
            task_file = os.path.join(cgroup_path, file_name)
            try:
                with open(task_file, "a", encoding="utf-8") as f:
                    f.write(f"{task_id}\n")
                    print(f"success to restore task {task_id} to {task_file}")
            except Exception as e:
                print(f"error: failed to restore task {task_id} to {task_file} -> {e}")
