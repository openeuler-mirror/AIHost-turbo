from collections import defaultdict

from a_sched.task.task_base import Task, PriorityLevel
from a_sched.task.thread_task import ThreadTask
from a_sched.task.process_task import ProcessTask
from a_sched.task.npu_task import NpuTask
import a_sched.utils as utils


class TaskGroup:
    def __init__(self, group_id: int, name: str = "", desc: str = ""):
        # 基本信息
        self.group_id: int = group_id
        self.name: str = name
        self.description: str = desc

        # 任务管理
        self.thread_tasks: dict[int, ThreadTask] = {}  # tid -> task
        self.process_tasks: dict[int, ProcessTask] = {}  # pid -> task
        self.npu_tasks: dict[int, NpuTask] = {}  # npu_id -> task

        # 调度管理
        self.cpus = utils.CPUMask()  # 分配到的cpu
        self.socket: list[int] = []  # 调度到的socket
        self.numa: list[int] = []  # 调度到的numa
        self.cluster: list[int] = []  # 调度到的cluster
        self.isolate_numa: list[int] = []  # 调度到的隔离numa
        self.isolate_cluster: list[int] = []  # 调度到的隔离cluster
        self.isolate_cpus = utils.CPUMask()  # 分配到的隔离cpu

    @property
    def all_tasks_num(self) -> int:
        return len(self.thread_tasks) + len(self.process_tasks) + len(self.npu_tasks)

    @property
    def all_tasks(self) -> list[Task]:
        all_tasks: list[Task] = []
        for _, process in self.process_tasks.items():
            all_tasks.append(process)
        for _, thread in self.thread_tasks.items():
            all_tasks.append(thread)
        for _, npu in self.npu_tasks.items():
            all_tasks.append(npu)
        return all_tasks

    def get_normal_prio_tasks(self) -> list[Task]:
        normal_prio_tasks: list[Task] = []
        for _, process in self.process_tasks.items():
            if process.priority == PriorityLevel.NORMAL:
                normal_prio_tasks.append(process)
        for _, thread in self.thread_tasks.items():
            if thread.priority == PriorityLevel.NORMAL:
                normal_prio_tasks.append(thread)
        return normal_prio_tasks

    def get_high_prio_tasks(self) -> list[Task]:
        high_prio_tasks: list[Task] = []
        for _, process in self.process_tasks.items():
            if process.priority == PriorityLevel.HIGH:
                high_prio_tasks.append(process)
        for _, thread in self.thread_tasks.items():
            # bind_npu的thread被加到npu_task中处理，这里不用再加了
            if thread.priority == PriorityLevel.HIGH and thread.bind_npu is None:
                high_prio_tasks.append(thread)
        for _, npu in self.npu_tasks.items():
            high_prio_tasks.append(npu)
        return high_prio_tasks

    def get_high_prio_threads(self) -> dict[int, list[int]]:
        high_prio_threads = defaultdict(list)
        for tid, thread in self.thread_tasks.items():
            if thread.priority == PriorityLevel.HIGH and thread.bind_npu is None:
                pid = utils.get_thread_pid_by_tid(tid)
                high_prio_threads[pid].append(tid)
        for npu in self.npu_tasks.values():
            npu_threads = npu.get_all_threads()
            for pid, threads in npu_threads.items():
                high_prio_threads[pid].extend(threads)
        return high_prio_threads

    def __str__(self) -> str:
        return (
            f"TaskGroup[{self.group_id}]: {f'name={self.name}, ' if self.name else ''} "
            f"socket={self.socket}, "
            f"numa=[{utils.compress_continuous(self.numa)}], "
            f"cluster=[{utils.compress_continuous(self.cluster)}], "
            f"cpu=[{self.cpus}]"
            f"{f', isol_numa=[{utils.compress_continuous(self.isolate_numa)}]' if self.isolate_numa else ''}"
            f"{f', isol_cluster=[{utils.compress_continuous(self.isolate_cluster)}]' if self.isolate_cluster else ''}"
            f"{f', isol_cpu=[{self.isolate_cpus}]' if self.isolate_cpus.count() != 0 else ''}"
        )
