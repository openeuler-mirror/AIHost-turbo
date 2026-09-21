from __future__ import annotations
from abc import ABC, abstractmethod

from a_sched.task.task_base import Task, TaskType, PriorityLevel
from a_sched.task.thread_task import ThreadTask
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils

ACL_THREAD = "acl_thread"
RELEASE_THREAD = "release_thread"
RT_RECYCLE_THREAD = "RT_RECYCLE"
SQ_IRQ = "sq_send_trigger_irq"
CQ_IRQ = "cq_update_irq"
TRS_MBOX_IRQ_PREFIX = "trs-mbox"


class NpuTask(Task, ABC):

    def __init__(self, task_type: TaskType, group_id: int, npu_id: int):
        super().__init__(task_type=task_type, group_id=group_id, task_id=npu_id)

        self.priority = PriorityLevel.HIGH  # npu为高优先级任务
        self.bind_pid: int | None = None  # npu关联的worker进程pid
        self.bind_high_prio_thread: list[ThreadTask] = [] # npu关联的高优先级线程

    @abstractmethod
    def init_npu(self) -> None:
        pass

    @abstractmethod
    def print(self) -> None:
        pass

    @property
    def processes(self) -> list[int]:
        return []

    def get_all_threads(self) -> dict[int, list[int]]:
        thread_list = [self._acl_thread, self._release_thread, self._rt_recycle_thread]
        thread_list.extend(thread.task_id for thread in self.bind_high_prio_thread)
        return {self.bind_pid: [tid for tid in thread_list if tid is not None]}


class NpuTaskA3(NpuTask):

    def __init__(self, group_id: int, npu_id: int):
        super().__init__(task_type=TaskType.NPU_A3, group_id=group_id, npu_id=npu_id)

        self._acl_thread: int | None = None  # acl线程 tid
        self._release_thread: int | None = None  # release线程 tid
        self._rt_recycle_thread: int | None = None  # rt_recycle线程 tid
        self._sq_irq: int | None = None  # sq_send_trigger_irq中断号
        self._cq_irqs: list[int] = []  # cq_update_irq中断号，有16个
        self._trs_mbox_irq: int | None = None  # trs_mbox中断号
        self._dev_sq_task: int | None = None  # dev_sq_task进程pid
        self._dev_sq_send_wq: int | None = None  # dev_sq_send_wq进程pid

        self._acl_thread_cpus: list[int] = []
        self._release_thread_cpus: list[int] = []
        self._rt_recycle_thread_cpus: list[int] = []
        self._sq_irq_cpus: list[int] = []
        self._cq_irqs_cpus: list[int] = []
        self._trs_mbox_irq_cpus: list[int] = []
        self._dev_sq_task_cpus: list[int] = []
        self._dev_sq_send_wq_cpus: list[int] = []

        self.trs_mbox_name = f"{TRS_MBOX_IRQ_PREFIX}-{self.task_id}-0"
        self.dev_sq_task_name = f"dev{self.task_id}_sq_task"
        self.dev_sq_send_wq_name = f"dev{self.task_id}_sq_send_wq"

    def init_npu(self) -> None:
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
        need_cpu += len(self.bind_high_prio_thread)
        return need_cpu

    @property
    def min_cluster(self) -> int:
        CPUS_PER_CLUSTER = 4
        base = self.min_cpu // CPUS_PER_CLUSTER
        extra = self.min_cpu % CPUS_PER_CLUSTER
        return base + 1 if extra > 0 else base

    @property
    def processes(self) -> list[int]:
        procs = []
        if self._dev_sq_task is not None:
            procs.append(self._dev_sq_task)
        if self._dev_sq_send_wq is not None:
            procs.append(self._dev_sq_send_wq)
        return procs

    @property
    def irqs(self) -> list[tuple[int, str]]:
        irqs = []
        if self._sq_irq is not None:
            irqs.append((self._sq_irq, SQ_IRQ))
        for cq in self._cq_irqs:
            irqs.append((cq, CQ_IRQ))
        if self._trs_mbox_irq is not None:
            irqs.append((self._trs_mbox_irq, self.trs_mbox_name))
        return irqs

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
        tids = utils.get_tid_by_thread_name(thread_name=ACL_THREAD, pid=self.bind_pid)
        if not tids:
            print(f"[Warning] cannot found {ACL_THREAD} for npu [{self.task_id}]")
        else:
            self._acl_thread = tids[0]
            print(f"get npu thread (name={ACL_THREAD}, tid={self._acl_thread}) success.")

        # release_thread
        tids = utils.get_tid_by_thread_name(thread_name=RELEASE_THREAD, pid=self.bind_pid)
        if not tids:
            print(f"[Warning] cannot found {RELEASE_THREAD} for npu [{self.task_id}]")
        else:
            self._release_thread = tids[0]
            print(f"get npu thread (name={RELEASE_THREAD}, tid={self._release_thread}) success.")

        # rt_recycle
        tids = utils.get_tid_by_thread_name(thread_name=RT_RECYCLE_THREAD, pid=self.bind_pid)
        if not tids:
            raise ValueError(f"[Warning] cannot found {RT_RECYCLE_THREAD} for npu [{self.task_id}]")
        else:
            self._rt_recycle_thread = tids[0]
            print(f"get npu thread (name={RT_RECYCLE_THREAD}, tid={self._rt_recycle_thread}) success.")

    def _get_npu_irqs(self) -> None:
        print(f"Starting get npu[{self.task_id}] irqs...")

        # sq
        irqs = utils.get_npu_irq_by_name(irq_name=SQ_IRQ, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {SQ_IRQ} of npu [{self.task_id}]")
        else:
            self._sq_irq = irqs[0]
            print(f"get npu irq (name={SQ_IRQ}, id={self._sq_irq}) success.")

        # cq
        irqs = utils.get_npu_irq_by_name(irq_name=CQ_IRQ, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {CQ_IRQ} of npu [{self.task_id}]")
        else:
            self._cq_irqs = irqs
            print(f"get npu irq (name={CQ_IRQ}, id={self._cq_irqs}) success.")

        # trs_mbox
        irqs = utils.get_npu_irq_by_name(irq_name=self.trs_mbox_name, npu_id=self.task_id)
        if not irqs:
            print(f"[Warning] cannot found {self.trs_mbox_name} of npu [{self.task_id}]")
        else:
            self._trs_mbox_irq = irqs[0]
            print(f"get npu irq (name={self.trs_mbox_name}, id={self._trs_mbox_irq}) success.")

    def bind_cpu(self) -> None:
        if self._acl_thread is not None and self._acl_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._acl_thread}]({ACL_THREAD}) "
                f"to cpu {self._acl_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._acl_thread, cpus=self._acl_thread_cpus)

        if self._release_thread is not None and self._release_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._release_thread}]({RELEASE_THREAD}) "
                f"to cpu {self._release_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._release_thread, cpus=self._release_thread_cpus)

        if self._rt_recycle_thread is not None and self._rt_recycle_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}) "
                f"to cpu {self._rt_recycle_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._rt_recycle_thread, cpus=self._rt_recycle_thread_cpus)

        if self._sq_irq is not None and self._sq_irq_cpus:
            print(
                f"binding npu[{self.task_id}] irq[{self._sq_irq}]({SQ_IRQ})"
                f"to cpu {self._sq_irq_cpus}"
            )
            utils.bind_irq_to_cpus(irq_id=self._sq_irq, cpus=self._sq_irq_cpus, irq_name=SQ_IRQ)

        if self._cq_irqs and self._cq_irqs_cpus:
            print(
                f"binding npu[{self.task_id}] irq{self._cq_irqs}({CQ_IRQ})"
                f"to cpu {self._cq_irqs_cpus}"
            )
            for cq in self._cq_irqs:
                utils.bind_irq_to_cpus(irq_id=cq, cpus=self._cq_irqs_cpus, irq_name=CQ_IRQ)

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
                f"[Error] assign cpu for npu[{self.task_id}] fail, "
                f"no enough cpus, assigned {cpu_num}, need {self.min_cpu}"
            )
            return

        start = 0
        for thread in self.bind_high_prio_thread:
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

    def print(self) -> None:
        print(f"  - {self}")
        if self._dev_sq_task is not None:
            print(
                f"    - Process[{self._dev_sq_task}]({self.dev_sq_task_name}): "
                f"cpu={utils.compress_continuous(self._dev_sq_task_cpus)}"
            )
        if self._dev_sq_send_wq is not None:
            print(
                f"    - Process[{self._dev_sq_send_wq}]({self.dev_sq_send_wq_name}): "
                f"cpu={utils.compress_continuous(self._dev_sq_send_wq_cpus)}"
            )
        if self._acl_thread is not None:
            print(
                f"    - Thread[{self._acl_thread}]({ACL_THREAD}): "
                f"cpu={utils.compress_continuous(self._acl_thread_cpus)}"
            )
        if self._release_thread is not None:
            (
                f"    - Thread[{self._release_thread}]({RELEASE_THREAD}): "
                f"cpu={utils.compress_continuous(self._release_thread_cpus)}"
            )
        if self._rt_recycle_thread is not None:
            print(
                f"    - Thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}): "
                f"cpu={utils.compress_continuous(self._rt_recycle_thread_cpus)}"
            )
        if self._sq_irq is not None:
            print(
                f"    - Irq[{self._sq_irq}]({SQ_IRQ}): "
                f"cpu={utils.compress_continuous(self._sq_irq_cpus)}"
            )
        if self._cq_irqs:
            print(
                f"    - Irq[{self._cq_irqs[0]}]({CQ_IRQ}): "
                f"cpu={utils.compress_continuous(self._cq_irqs_cpus)}"
            )
        if self._trs_mbox_irq is not None:
            print(
                f"    - Irq[{self._trs_mbox_irq}]({self.trs_mbox_name}): "
                f"cpu={utils.compress_continuous(self._trs_mbox_irq_cpus)}"
            )

    def print_actual_affinity(self, domain: AffinityDomainManager) -> None:
        print(f"  - NPU_A3[{self.task_id}]: ")
        if self._dev_sq_task is not None:
            cpus = utils.get_process_cpus_by_pid(pid=self._dev_sq_task)
            print(
                f"    - Process[{self._dev_sq_task}]({self.dev_sq_task_name}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._dev_sq_send_wq is not None:
            cpus = utils.get_npu_work_queue_cpus_by_name(wq_name=self.dev_sq_send_wq_name)
            print(
                f"    - Process[{self._dev_sq_send_wq}]({self.dev_sq_send_wq_name}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._acl_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._acl_thread)
            print(
                f"    - Thread[{self._acl_thread}]({ACL_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._release_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._release_thread)
            print(
                f"    - Thread[{self._release_thread}]({RELEASE_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._rt_recycle_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._rt_recycle_thread)
            print(
                f"    - Thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._sq_irq is not None:
            cpus = utils.get_irq_cpus_by_irq_id(irq_id=self._sq_irq)
            print(
                f"    - Irq[{self._sq_irq}]({SQ_IRQ}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        for cq in self._cq_irqs:
            cpus = utils.get_irq_cpus_by_irq_id(irq_id=cq)
            print(
                f"    - Irq[{cq}]({CQ_IRQ}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._trs_mbox_irq is not None:
            cpus = utils.get_irq_cpus_by_irq_id(irq_id=self._trs_mbox_irq)
            print(
                f"    - Irq[{self._trs_mbox_irq}]({self.trs_mbox_name}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )

    def build_dev_bind_data(self) -> dict:
        bind_data = {}
        if self._dev_sq_task is not None:
            try:
                bind_data["dev_sq_task"] = {
                    "pid": self._dev_sq_task,
                    "name": self.dev_sq_task_name,
                    "cpu_affinity": utils.get_process_cpus_by_pid(pid=self._dev_sq_task),
                }
            except Exception as e:
                print(f"npu[{self.task_id}] {self.dev_sq_task_name} get cpu affinity failed, {str(e)}")
        if self._dev_sq_send_wq is not None:
            try:
                bind_data["dev_sq_send_wq"] = {
                    "wq_name": self.dev_sq_send_wq_name,
                    "cpu_affinity": utils.get_npu_work_queue_cpus_by_name(wq_name=self.dev_sq_send_wq_name),
                }
            except Exception as e:
                print(f"npu[{self.task_id}] {self.dev_sq_send_wq_name} get cpu affinity failed, {str(e)}")
        return bind_data


class NpuTaskA5(NpuTask):

    def __init__(self, group_id: int, npu_id: int):
        super().__init__(task_type=TaskType.NPU_A5, group_id=group_id, npu_id=npu_id)

        self._acl_thread: int | None = None  # acl线程 tid
        self._release_thread: int | None = None  # release线程 tid
        self._rt_recycle_thread: int | None = None  # rt_recycle线程 tid

        self._acl_thread_cpus: list[int] = []
        self._release_thread_cpus: list[int] = []
        self._rt_recycle_thread_cpus: list[int] = []

    def init_npu(self) -> None:
        print(f"\nStarting init npu[{self.task_id}]...")
        self._get_npu_threads()

    @property
    def min_cpu(self) -> int:
        need_cpu = 0
        need_cpu += 1 if self._acl_thread is not None else 0
        need_cpu += 1 if self._release_thread is not None else 0
        need_cpu += 1 if self._rt_recycle_thread is not None else 0
        need_cpu += len(self.bind_high_prio_thread)
        return need_cpu

    @property
    def min_cluster(self) -> int:
        CPUS_PER_CLUSTER = 8
        base = self.min_cpu // CPUS_PER_CLUSTER
        extra = self.min_cpu % CPUS_PER_CLUSTER
        return base + 1 if extra > 0 else base

    def _get_npu_threads(self) -> None:
        print(f"Starting get npu[{self.task_id}] threads...")

        # acl_thread
        tids = utils.get_tid_by_thread_name(thread_name=ACL_THREAD, pid=self.bind_pid)
        if not tids:
            print(f"[Warning] cannot found {ACL_THREAD} for npu [{self.task_id}]")
        else:
            self._acl_thread = tids[0]
            print(f"get npu thread (name={ACL_THREAD}, tid={self._acl_thread}) success.")

        # release_thread
        tids = utils.get_tid_by_thread_name(thread_name=RELEASE_THREAD, pid=self.bind_pid)
        if not tids:
            print(f"[Warning] cannot found {RELEASE_THREAD} for npu [{self.task_id}]")
        else:
            self._release_thread = tids[0]
            print(f"get npu thread (name={RELEASE_THREAD}, tid={self._release_thread}) success.")

        # rt_recycle
        tids = utils.get_tid_by_thread_name(thread_name=RT_RECYCLE_THREAD, pid=self.bind_pid)
        if not tids:
            raise ValueError(f"[Warning] cannot found {RT_RECYCLE_THREAD} for npu [{self.task_id}]")
        else:
            self._rt_recycle_thread = tids[0]
            print(f"get npu thread (name={RT_RECYCLE_THREAD}, tid={self._rt_recycle_thread}) success.")

    def bind_cpu(self) -> None:
        if self._acl_thread is not None and self._acl_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._acl_thread}]({ACL_THREAD}) "
                f"to cpu {self._acl_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._acl_thread, cpus=self._acl_thread_cpus)

        if self._release_thread is not None and self._release_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._release_thread}]({RELEASE_THREAD}) "
                f"to cpu {self._release_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._release_thread, cpus=self._release_thread_cpus)

        if self._rt_recycle_thread is not None and self._rt_recycle_thread_cpus:
            print(
                f"binding npu[{self.task_id}] thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}) "
                f"to cpu {self._rt_recycle_thread_cpus}"
            )
            utils.bind_thread_to_cpus(tid=self._rt_recycle_thread, cpus=self._rt_recycle_thread_cpus)

    def assign_cpu(self, cpus: list[int]) -> None:
        self.cpus.set_list(cpus)
        cpu_num = len(cpus)
        if cpu_num < self.min_cpu:
            print(
                f"[Error] assign cpu for npu[{self.task_id}] fail, "
                f"no enough cpus, assigned {cpu_num}, need {self.min_cpu}"
            )
            return

        start = 0
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
        for thread in self.bind_high_prio_thread:
            end = start + 1
            thread.assign_cpu(cpus[start:end])
            thread.socket = list(self.socket)
            thread.numa = list(self.numa)
            thread.cluster = list(self.cluster)
            start = end

    def print(self) -> None:
        print(f"  - {self}")
        if self._acl_thread is not None:
            print(
                f"    - Thread[{self._acl_thread}]({ACL_THREAD}): "
                f"cpu={utils.compress_continuous(self._acl_thread_cpus)}"
            )
        if self._release_thread is not None:
            print(
                f"    - Thread[{self._release_thread}]({RELEASE_THREAD}): "
                f"cpu={utils.compress_continuous(self._release_thread_cpus)}"
            )
        if self._rt_recycle_thread is not None:
            print(
                f"    - Thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}): "
                f"cpu={utils.compress_continuous(self._rt_recycle_thread_cpus)}"
            )

    def print_actual_affinity(self, domain: AffinityDomainManager) -> None:
        print(f"  - NPU_A5[{self.task_id}]: ")
        if self._acl_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._acl_thread)
            print(
                f"    - Thread[{self._acl_thread}]({ACL_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._release_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._release_thread)
            print(
                f"    - Thread[{self._release_thread}]({RELEASE_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )
        if self._rt_recycle_thread is not None:
            cpus = utils.get_thread_cpus_by_tid(tid=self._rt_recycle_thread)
            print(
                f"    - Thread[{self._rt_recycle_thread}]({RT_RECYCLE_THREAD}): "
                f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
                f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
                f"cpu=[{utils.CPUMask().from_list(cpus)}]"
            )


def create_npu_task(group_id: int, npu_id: int) -> NpuTask:
    device_type = utils.get_ascend_device_type()
    if device_type == utils.AscendDeviceType.A3:
        return NpuTaskA3(group_id=group_id, npu_id=npu_id)
    elif device_type == utils.AscendDeviceType.A5:
        return NpuTaskA5(group_id=group_id, npu_id=npu_id)
    else:
        raise RuntimeError(f"Unsupported ascend device type {device_type}")
