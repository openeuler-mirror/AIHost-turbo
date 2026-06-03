from __future__ import annotations
from collections import defaultdict
from enum import Enum, auto
from datetime import datetime

from a_sched.utils import CPUMask
import a_sched.utils as utils


class PriorityLevel(Enum):
    NORMAL = auto()
    HIGH = auto()


class TaskType(Enum):
    THREAD = auto()  # 线程
    PROCESS = auto()  # 进程
    NPU = auto()  # NPU


class TaskGroupStatus(Enum):
    UNKNOWN = auto()
    CREATED = auto()


class Task:
    def __init__(self, task_type: TaskType, group_id: int, task_id: int, name: str | None = None):
        # 基本信息
        self.group_id: int = group_id
        self.task_id: int = task_id
        self.name: str | None = name
        self.task_type: TaskType = task_type
        self.priority: PriorityLevel = PriorityLevel.NORMAL  # 默认优先级为Normal
        self.bind_npu: int | None = None

        # 调度管理
        self.cpus: CPUMask = CPUMask()  # 分配到的cpu
        self.socket: list[int] = []  # 调度到的socket
        self.numa: list[int] = []  # 调度到的numa
        self.cluster: list[int] = []  # 调度到的cluster
        self.do_isolate: bool = False  # 是否做了隔离调度

        self._type_names = {TaskType.THREAD: "THREAD", TaskType.PROCESS: "PROCESS", TaskType.NPU: "NPU"}
        self._priority_names = {PriorityLevel.NORMAL: "NORMAL", PriorityLevel.HIGH: "HIGH"}

    @property
    def min_cpu(self) -> int:
        return 1

    @property
    def min_cluster(self) -> int:
        return 1

    def assign_cpu(self, cpus: list[int]) -> None:
        pass

    def bind_cpu(self) -> None:
        pass

    def __str__(self) -> str:
        type_name = self._type_names.get(self.task_type, "UNKNOWN")
        priority = self._priority_names.get(self.priority, "UNKNOWN")
        return (
            f"{type_name}[{self.task_id}]: "
            f"name={self.name if self.name is not None else ''}, priority={priority}, "
            f"socket={self.socket}, "
            f"numa=[{utils.compress_continuous(self.numa)}], "
            f"cluster=[{utils.compress_continuous(self.cluster)}], "
            f"cpu=[{self.cpus}]"
        )


class ThreadTask(Task):
    def __init__(self, group_id: int, tid: int, pid: int, name: str | None = None):
        super().__init__(TaskType.THREAD, group_id=group_id, task_id=tid, name=name)
        self.pid: int = pid

    def bind_cpu(self) -> None:
        print(f"binding thread[{self.task_id}]({self.name}) to cpu [{self.cpus}]")
        utils.bind_thread_to_cpus(tid=self.task_id, cpus=self.cpus.to_list())

    def assign_cpu(self, cpus: list[int]) -> None:
        if not cpus:
            print(f"[Error] assign cpu for thread[{self.task_id}](name={self.name}) fail, cpus={cpus}")
            return
        self.cpus.set_list(cpus[:1])


class ProcessTask(Task):
    def __init__(self, group_id: int, pid: int, name: str | None = None):
        super().__init__(TaskType.PROCESS, group_id=group_id, task_id=pid, name=name)

    def bind_cpu(self) -> None:
        print(f"binding process[{self.task_id}]({self.name}) to cpu [{self.cpus}]")
        utils.bind_process_to_cpus(pid=self.task_id, cpus=self.cpus.to_list())

    def assign_cpu(self, cpus: list[int]) -> None:
        self.cpus.set_list(cpus)


class NpuTask(Task):
    ACL_THREAD = "acl_thread"
    RELEASE_THREAD = "release_thread"
    RT_RECYCLE_THREAD = "RT_RECYCLE"
    SQ_IRQ = "sq_send_trigger_irq"
    CQ_IRQ = "cq_update_irq"
    TRS_MBOX_IRQ_PREFIX = "trs-mbox"

    def __init__(self, group_id: int, npu_id: int, bind_pid: int, name: str = ""):
        super().__init__(TaskType.NPU, group_id=group_id, task_id=npu_id, name=name)
        self.priority = PriorityLevel.HIGH

        self._bind_pid = bind_pid  # npu关联的worker进程pid
        self._acl_thread: int | None = None  # acl线程 tid
        self._release_thread: int | None = None  # release线程 tid
        self._rt_recycle_thread: int | None = None  # rt_recycle线程 tid
        self._sq_irq: int | None = None  # sq_send_trigger_irq中断号
        self._cq_irqs: list[int] = []  # cq_update_irq中断号，有16个
        self._trs_mbox_irq: int | None = None  # trs_mbox中断号
        self._dev_sq_task: int | None = None  # dev_sq_task进程pid
        self._dev_sq_send_wq: int | None = None  # dev_sq_send_wq进程pid

        self._bind_high_prio_thread: list[ThreadTask] = []

        self.trs_mbox_name = f"{self.TRS_MBOX_IRQ_PREFIX}-{self.task_id}-0"
        self.dev_sq_task_name = f"dev{self.task_id}_sq_task"
        self.dev_sq_send_wq_name = f"dev{self.task_id}_sq_send_wq"

        self._init_npu()

        self._acl_thread_cpus: list[int] = []
        self._release_thread_cpus: list[int] = []
        self._rt_recycle_thread_cpus: list[int] = []
        self._sq_irq_cpus: list[int] = []
        self._cq_irqs_cpus: list[int] = []
        self._trs_mbox_irq_cpus: list[int] = []
        self._dev_sq_task_cpus: list[int] = []
        self._dev_sq_send_wq_cpus: list[int] = []

    def _init_npu(self) -> None:
        print(f"\nStarting init npu[{self.task_id}]...")
        self._get_npu_threads()
        self._get_npu_irqs()
        self._get_npu_dev_sq()

    @property
    def min_cpu(self) -> int:
        need_cpu = 0
        need_cpu += 1 if self._acl_thread is not None else 0
        need_cpu += 1 if self._release_thread is not None else 0
        need_cpu += 1 if self._rt_recycle_thread is not None else 0
        need_cpu += 1 if self._sq_irq is not None else 0
        need_cpu += 1 if self._cq_irqs else 0
        need_cpu += 1 if self._trs_mbox_irq is not None else 0
        need_cpu += 1 if self._dev_sq_task is not None else 0
        need_cpu += 1 if self._dev_sq_send_wq is not None else 0
        need_cpu += len(self._bind_high_prio_thread)
        return need_cpu

    @property
    def min_cluster(self) -> int:
        # todo: 需要根据实际硬件参数来计算cluster数量，暂时按照a3来处理
        cpu_num_per_cluster = 4
        base = self.min_cpu // cpu_num_per_cluster
        extra = self.min_cpu % cpu_num_per_cluster
        return base + 1 if extra > 0 else base

    def _get_npu_dev_sq(self) -> None:
        print(f"Starting get npu[{self.task_id}] dev_sq...")

        # dev_sq_task
        pids = utils.get_pid_by_process_name(process_name=self.dev_sq_task_name)
        if not pids:
            print(f"[Warning] cannot found {self.dev_sq_task_name} for npu [{self.task_id}]")
        else:
            self._dev_sq_task = pids[0][0]
            print(f"get npu process (name={self.dev_sq_task_name}, pid={self._dev_sq_task}) success.")

        # dev_sq_send_wq
        pids = utils.get_pid_by_process_name(process_name=self.dev_sq_send_wq_name[:15])
        if not pids:
            print(f"[Warning] cannot found {self.dev_sq_send_wq_name} for npu [{self.task_id}]")
        else:
            self._dev_sq_send_wq = pids[0][0]
            print(f"get npu process (name={self.dev_sq_send_wq_name}, pid={self._dev_sq_send_wq}) success.")

    def _get_npu_threads(self) -> None:
        print(f"Starting get npu[{self.task_id}] threads...")

        # acl_thread
        tids = utils.get_tid_by_thread_name(thread_name=self.ACL_THREAD, pid=self._bind_pid)
        if not tids:
            print(f"[Warning] cannot found {self.ACL_THREAD} for npu [{self.task_id}]")
        else:
            self._acl_thread = tids[0]
            print(f"get npu thread (name={self.ACL_THREAD}, tid={self._acl_thread}) success.")

        # release_thread
        tids = utils.get_tid_by_thread_name(thread_name=self.RELEASE_THREAD, pid=self._bind_pid)
        if not tids:
            print(f"[Warning] cannot found {self.RELEASE_THREAD} for npu [{self.task_id}]")
        else:
            self._release_thread = tids[0]
            print(f"get npu thread (name={self.RELEASE_THREAD}, tid={self._release_thread}) success.")

        # rt_recycle
        tids = utils.get_tid_by_thread_name(thread_name=self.RT_RECYCLE_THREAD, pid=self._bind_pid)
        if not tids:
            raise ValueError(f"[Warning] cannot found {self.RT_RECYCLE_THREAD} for npu [{self.task_id}]")
        else:
            self._rt_recycle_thread = tids[0]
            print(f"get npu thread (name={self.RT_RECYCLE_THREAD}, tid={self._rt_recycle_thread}) success.")

    def _get_npu_irqs(self) -> None:
        print(f"Starting get npu[{self.task_id}] irqs...")

        # sq
        irqs = utils.get_npu_irq_by_name(irq_name=self.SQ_IRQ, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {self.SQ_IRQ} of npu [{self.task_id}]")
        else:
            self._sq_irq = irqs[0]
            print(f"get npu irq (name={self.SQ_IRQ}, id={self._sq_irq}) success.")

        # cq
        irqs = utils.get_npu_irq_by_name(irq_name=self.CQ_IRQ, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {self.CQ_IRQ} of npu [{self.task_id}]")
        else:
            self._cq_irqs = irqs
            print(f"get npu irq (name={self.CQ_IRQ}, id={self._cq_irqs}) success.")

        # trs_mbox
        irqs = utils.get_npu_irq_by_name(irq_name=self.trs_mbox_name, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {self.trs_mbox_name} of npu [{self.task_id}]")
        else:
            self._trs_mbox_irq = irqs[0]
            print(f"get npu irq (name={self.trs_mbox_name}, id={self._trs_mbox_irq}) success.")

    def bind_cpu(self) -> None:
        for task in self._bind_high_prio_thread:
            task.bind_cpu()

        if self._acl_thread is not None and self._acl_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._acl_thread}]({self.ACL_THREAD}) "
                f"to cpu {self._acl_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._acl_thread, cpus=self._acl_thread_cpus)

        if self._release_thread is not None and self._release_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._release_thread}]({self.RELEASE_THREAD}) "
                f"to cpu {self._release_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._release_thread, cpus=self._release_thread_cpus)

        if self._rt_recycle_thread is not None and self._rt_recycle_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._rt_recycle_thread}]({self.RT_RECYCLE_THREAD}) "
                f"to cpu {self._rt_recycle_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._rt_recycle_thread, cpus=self._rt_recycle_thread_cpus)

        if self._sq_irq is not None and self._sq_irq_cpus:
            print(f"binding npu[{self.task_id}] irq[{self._sq_irq}]({self.SQ_IRQ}) to cpu {self._sq_irq_cpus}")
            utils.bind_irq_to_cpus(irq_id=self._sq_irq, cpus=self._sq_irq_cpus, irq_name=self.SQ_IRQ)

        if self._cq_irqs and self._cq_irqs_cpus:
            print(f"binding npu[{self.task_id}] irq{self._cq_irqs}({self.CQ_IRQ}) to cpu {self._cq_irqs_cpus}")
            for cq in self._cq_irqs:
                utils.bind_irq_to_cpus(irq_id=cq, cpus=self._cq_irqs_cpus, irq_name=self.CQ_IRQ)

        if self._trs_mbox_irq is not None and self._trs_mbox_irq_cpus:
            print(
                f"binding npu[{self.task_id}] irq[{self._trs_mbox_irq}]({self.trs_mbox_name}) "
                f"to cpu {self._trs_mbox_irq_cpus}"
            )
            utils.bind_irq_to_cpus(irq_id=self._trs_mbox_irq, cpus=self._trs_mbox_irq_cpus, irq_name=self.trs_mbox_name)

        if self._dev_sq_task is not None and self._dev_sq_task_cpus:
            print(
                f"binding npu[{self.task_id}] process[{self._dev_sq_task}]({self.dev_sq_task_name}) "
                f"to cpu {self._dev_sq_task_cpus}"
            )
            utils.bind_process_to_cpus(pid=self._dev_sq_task, cpus=self._dev_sq_task_cpus)

        if self._dev_sq_send_wq is not None and self._dev_sq_send_wq_cpus:
            print(
                f"binding npu[{self.task_id}] process[{self._dev_sq_send_wq}]({self.dev_sq_send_wq_name}) "
                f"to cpu {self._dev_sq_send_wq_cpus}"
            )
            utils.bind_npu_sq_send_wq_to_cpus(npu_id=self.task_id, cpus=self._dev_sq_send_wq_cpus)

    def assign_cpu(self, cpus: list[int]) -> None:
        self.cpus.set_list(cpus)
        cpu_num = len(cpus)
        if cpu_num < self.min_cpu:
            print(
                f"[Error] assign cpu for npu[{self.task_id}] fail, no enough cpus, assigned {cpu_num}, need {self.min_cpu}"
            )
            return

        start = 0
        for thread in self._bind_high_prio_thread:
            end = start + 1
            thread.assign_cpu(cpus[start:end])
            start = end
            thread.socket = list(self.socket)
            thread.numa = list(self.numa)
            thread.cluster = list(self.cluster)
        if self._acl_thread is not None:
            end = start + 1
            self._acl_thread_cpus = cpus[start:end]
            start = end
        if self._release_thread is not None:
            end = start + 1
            self._release_thread_cpus = cpus[start:end]
            start = end
        if self._rt_recycle_thread is not None:
            end = start + 1
            self._rt_recycle_thread_cpus = cpus[start:end]
            start = end
        if self._dev_sq_task is not None:
            end = start + 1
            self._dev_sq_task_cpus = cpus[start:end]
            start = end
        if self._dev_sq_send_wq is not None:
            end = start + 1
            self._dev_sq_send_wq_cpus = cpus[start:end]
            start = end
        if self._sq_irq is not None:
            end = start + 1
            self._sq_irq_cpus = cpus[start:end]
            start = end
        if self._cq_irqs is not None:
            end = start + 1
            self._cq_irqs_cpus = cpus[start:end]
            start = end
        if self._trs_mbox_irq is not None:
            end = start + 1
            self._trs_mbox_irq_cpus = cpus[start:end]
            start = end

    def get_all_threads(self) -> dict[int, list[int]]:
        thread_list = [self._acl_thread, self._release_thread, self._rt_recycle_thread]
        thread_list.extend([thread.task_id for thread in self._bind_high_prio_thread])
        return {self._bind_pid: [tid for tid in thread_list if tid is not None]}

    def __str__(self) -> str:
        npu_str = ""
        type_name = self._type_names.get(self.task_type, "UNKNOWN")
        priority = self._priority_names.get(self.priority, "UNKNOWN")
        npu_str = (
            f"{type_name}[{self.task_id}]: priority={priority}, "
            f"socket={self.socket}, numa=[{utils.compress_continuous(self.numa)}], cluster=[{utils.compress_continuous(self.cluster)}], "
            f"cpu=[{self.cpus}]"
        )
        npu_str += (
            (
                f"\n    - Process[{self._dev_sq_task}]({self.dev_sq_task_name}): cpu={utils.compress_continuous(self._dev_sq_task_cpus)}"
            )
            if self._dev_sq_task is not None
            else ""
        )
        npu_str += (
            (
                f"\n    - Process[{self._dev_sq_send_wq}]({self.dev_sq_send_wq_name}): cpu={utils.compress_continuous(self._dev_sq_send_wq_cpus)}"
            )
            if self._dev_sq_send_wq is not None
            else ""
        )
        npu_str += (
            (
                f"\n    - Thread[{self._acl_thread}]({self.ACL_THREAD}): cpu={utils.compress_continuous(self._acl_thread_cpus)}"
            )
            if self._acl_thread is not None
            else ""
        )
        npu_str += (
            (
                f"\n    - Thread[{self._release_thread}]({self.RELEASE_THREAD}): cpu={utils.compress_continuous(self._release_thread_cpus)}"
            )
            if self._release_thread is not None
            else ""
        )
        npu_str += (
            (
                f"\n    - Thread[{self._rt_recycle_thread}]({self.RT_RECYCLE_THREAD}): cpu={utils.compress_continuous(self._rt_recycle_thread_cpus)}"
            )
            if self._rt_recycle_thread is not None
            else ""
        )
        npu_str += (
            (f"\n    - Irq[{self._sq_irq}]({self.SQ_IRQ}): cpu={utils.compress_continuous(self._sq_irq_cpus)}")
            if self._sq_irq is not None
            else ""
        )
        npu_str += (
            (f"\n    - Irq[{self._cq_irqs[0]}]({self.CQ_IRQ}): cpu={utils.compress_continuous(self._cq_irqs_cpus)}")
            if self._cq_irqs
            else ""
        )
        npu_str += (
            (
                f"\n    - Irq[{self._trs_mbox_irq}]({self.trs_mbox_name}): cpu={utils.compress_continuous(self._trs_mbox_irq_cpus)}"
            )
            if self._trs_mbox_irq is not None
            else ""
        )
        return npu_str


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
        self.cpus: CPUMask = CPUMask()  # 分配到的cpu
        self.socket: list[int] = []  # 调度到的socket
        self.numa: list[int] = []  # 调度到的numa
        self.cluster: list[int] = []  # 调度到的cluster
        self.isolate_numa: list[int] = []
        self.isolate_cluster: list[int] = []
        self.isolate_cpus: CPUMask = CPUMask()

        # 状态和统计
        self.created_at: datetime = datetime.now()
        self.status: TaskGroupStatus = TaskGroupStatus.CREATED

    def get_all_tasks_num(self) -> int:
        return len(self.thread_tasks) + len(self.process_tasks) + len(self.npu_tasks)

    def get_all_tasks(self) -> list[Task]:
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
            f"TaskGroup[{self.group_id}]: name={self.name}, "
            f"socket={self.socket}, "
            f"numa=[{utils.compress_continuous(self.numa)}], "
            f"cluster=[{utils.compress_continuous(self.cluster)}], "
            f"cpu=[{self.cpus}]"
            f"{f', isol_numa=[{utils.compress_continuous(self.isolate_numa)}]' if self.isolate_numa else ''}"
            f"{f', isol_cluster=[{utils.compress_continuous(self.isolate_cluster)}]' if self.isolate_cluster else ''}"
            f"{f', isol_cpu=[{self.isolate_cpus}]' if self.isolate_cpus.count() != 0 else ''}"
        )


class TaskManager:
    """调度任务管理"""

    def __init__(self) -> None:
        self.groups: dict[int, TaskGroup] = {}  # group_id -> task group
        self.process_to_npu: dict[int, int] = {}  # pid -> npu_id
        self.background_processes: dict[int, str] = {}  #  背景进程
        self.background_tasks_cpus: list[int] = []  # 背景任务分配的CPU
        self._current_group_index: int = 0

    def group_create(self, name: str = "") -> int:
        group_id = self._current_group_index
        self.groups[group_id] = TaskGroup(group_id=group_id, name=name)
        self._current_group_index += 1
        return group_id

    def destory_group(self, group_id: int) -> None:
        self.groups.pop(group_id)

    def group_add_thread(
        self,
        group_id: int,
        tid: int | None = None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("add thread failed, either tid or thread_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"add thread failed, group (id={group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"add thread failed, process (pid={pid}, name={process_name}) not found")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"add thread failed, thread (tid={tid}, name={thread_name}) not found")

        group.thread_tasks[tid] = ThreadTask(group_id=group_id, tid=tid, pid=pid, name=thread_name)

    def group_remove_thread(
        self,
        group_id: int,
        tid: int | None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("remove thread failed, either tid or thread_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"remove thread failed, group({group_id}) not found")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"remove thread failed, thread (tid={tid}, name={thread_name}) not found")

        group.thread_tasks.pop(tid, None)

    def group_add_process(
        self, group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("add process failed, either pid or process_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"add process failed, group (id={group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"add process failed, process (pid={pid}, name={process_name}) not found")

        group.process_tasks[pid] = ProcessTask(group_id=group_id, pid=pid, name=process_name)

    def group_remove_process(
        self, group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("remove process failed, either pid or process_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"remove process failed, group({group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"remove process failed, process (pid={pid}, name={process_name}) not found")

        group.process_tasks.pop(pid)

    def thread_set_high_priority(
        self,
        tid: int | None = None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("set high priority failed, either tid or thread_name must be set")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"set high priority failed, thread (tid={tid}, name={thread_name}) not found")

        for _, group in self.groups.items():
            task = group.thread_tasks.get(tid)
            if task is not None:
                task.priority = PriorityLevel.HIGH
                break

    def process_bind_npu(
        self, npu_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("process bind npu failed, either pid or process_name must be set")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"process bind npu failed, process (pid={pid}, name={process_name}) not found")

        group = self.find_group_by_pid(pid)
        if group is None:
            raise ValueError(f"process bind npu failed, not found group of process (pid={pid}, name={process_name})")

        self.process_to_npu[pid] = npu_id
        group.npu_tasks[npu_id] = NpuTask(group_id=group.group_id, npu_id=npu_id, bind_pid=pid, name=f"npu[{npu_id}]")

    def get_group_num(self) -> int:
        return len(self.groups)

    def get_all_group_id(self) -> list[int]:
        task_groups_id: list[int] = []
        for group_id in self.groups:
            task_groups_id.append(group_id)
        return task_groups_id

    def get_group(self, group_id: int) -> TaskGroup | None:
        return self.groups.get(group_id, None)

    def find_group_by_pid(self, find_pid: int) -> TaskGroup | None:
        for _, group in self.groups.items():
            for pid in group.process_tasks:
                if pid == find_pid:
                    return group
        return None

    def get_high_prio_tasks_of_group(self, group_id: int) -> list[Task]:
        group = self.get_group(group_id)
        return group.get_high_prio_tasks() if group is not None else []

    def print_all(self) -> None:
        for _, group in self.groups.items():
            self.print_task_group(group)

    def print_task_group(self, group: TaskGroup) -> None:
        print(f"{group}")
        for _, process in group.process_tasks.items():
            print(f"  - {process}")
        for _, thread in group.thread_tasks.items():
            print(f"  - {thread}")
        for _, npu in group.npu_tasks.items():
            print(f"  - {npu}")

    def update_high_prio_thread_bind_npu(self) -> None:
        for _, group in self.groups.items():
            for _, thread in group.thread_tasks.items():
                if thread.priority != PriorityLevel.HIGH:
                    continue
                thread.bind_npu = self.process_to_npu.get(thread.pid)
                if thread.bind_npu is None:
                    continue
                npu_task = group.npu_tasks.get(thread.bind_npu)
                if npu_task is not None:
                    npu_task._bind_high_prio_thread.append(thread)

    def scan_background_tasks(self) -> None:
        """
        扫描背景任务
        """

        user_pids = set()
        for _, group in self.groups.items():
            for pid in group.process_tasks.keys():
                user_pids.add(pid)
            for _, npu_task in group.npu_tasks.items():
                if npu_task._dev_sq_task is not None:
                    user_pids.add(npu_task._dev_sq_task)
                if npu_task._dev_sq_send_wq is not None:
                    user_pids.add(npu_task._dev_sq_send_wq)

        background_tasks = utils.get_all_user_processes()
        for pid, name in background_tasks:
            if pid not in user_pids:
                self.background_processes[pid] = name

        print(f"[BackgroundTask] Scanned {len(self.background_processes)} processes")
