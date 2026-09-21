from __future__ import annotations

from a_sched.task.task_base import Task, TaskType
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils


class ThreadTask(Task):
    def __init__(self, group_id: int, tid: int, pid: int, name: str | None = None):
        super().__init__(TaskType.THREAD, group_id=group_id, task_id=tid, name=name)
        self.pid: int = pid  # 线程所属的进程pid

    def bind_cpu(self) -> None:
        print(f"binding thread[{self.task_id}]({self.name}) to cpu [{self.cpus}]")
        utils.bind_thread_to_cpus(tid=self.task_id, cpus=self.cpus.to_list())

    def assign_cpu(self, cpus: list[int]) -> None:
        if not cpus:
            print(f"[Error] assign cpu for thread[{self.task_id}](name={self.name}) fail, cpus={cpus}")
            return
        self.cpus.set_list(cpus[:1])

    def print_actual_affinity(self, domain: AffinityDomainManager) -> None:
        priority = self._priority_names.get(self.priority, "UNKNOWN")
        cpus = utils.get_thread_cpus_by_tid(tid=self.task_id)
        print(
            f"  - THREAD[{self.task_id}]: name={self.name}, priority={priority}, "
            f"socket={domain.get_sockets_of_cpus(cpus)}, "
            f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
            f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
            f"cpu=[{utils.CPUMask().from_list(cpus)}]"
        )
