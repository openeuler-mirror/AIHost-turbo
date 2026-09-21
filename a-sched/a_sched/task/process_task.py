from __future__ import annotations

from a_sched.task.task_base import Task, TaskType
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils


class ProcessTask(Task):
    def __init__(self, group_id: int, pid: int, name: str | None = None):
        super().__init__(TaskType.PROCESS, group_id=group_id, task_id=pid, name=name)

    def bind_cpu(self) -> None:
        print(f"binding process[{self.task_id}]({self.name}) to cpu [{self.cpus}]")
        utils.bind_process_to_cpus(pid=self.task_id, cpus=self.cpus.to_list())

    def assign_cpu(self, cpus: list[int]) -> None:
        self.cpus.set_list(cpus)

    def print_actual_affinity(self, domain: AffinityDomainManager) -> None:
        priority = self._priority_names.get(self.priority, "UNKNOWN")
        cpus = utils.get_process_cpus_by_pid(pid=self.task_id)
        print(
            f"  - PROCESS[{self.task_id}]: name={self.name}, priority={priority}, "
            f"socket={domain.get_sockets_of_cpus(cpus)}, "
            f"numa=[{utils.compress_continuous(domain.get_numas_of_cpus(cpus))}], "
            f"cluster=[{utils.compress_continuous(domain.get_clusters_of_cpus(cpus))}], "
            f"cpu=[{utils.CPUMask().from_list(cpus)}]"
        )
