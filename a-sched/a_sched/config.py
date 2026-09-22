import a_sched.utils as utils

ISOL_AUTO = "auto"
ISOL_NUMA = "numa"
ISOL_CLUSTER = "cluster"


class AffinityConfig:

    def __init__(self) -> None:
        self.exclude_cpus: list[int] = []
        self.enable_cpuset = False
        self.isolate: str = ISOL_AUTO

    def set_exclude_cpu(self, cpu_str: str) -> None:
        self.exclude_cpus = utils.parse_cpu_affinity_string(cpu_str)
