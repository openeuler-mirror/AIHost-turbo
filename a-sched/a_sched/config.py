import a_sched.utils as utils

ISOL_AUTO = "auto"
ISOL_NUMA = "numa"
ISOL_CLUSTER = "cluster"

NPU_PROCESS_CLUSTER_MODE_ISOLATED = "isolated"
NPU_PROCESS_CLUSTER_MODE_COLOCATED = "colocated"
NPU_PROCESS_CLUSTER_MODE_SHARED = "shared"


class AffinityConfig:

    def __init__(self) -> None:
        self.exclude_cpus: list[int] = []
        self.enable_cpuset = False
        self.isolate: str = ISOL_AUTO
        self.drop_caches: bool = False
        self.enable_npu_topo_affinity: bool = False
        self.npu_process_cluster_mode: str = NPU_PROCESS_CLUSTER_MODE_ISOLATED

    def set_exclude_cpu(self, cpu_str: str) -> None:
        self.exclude_cpus = utils.parse_cpu_affinity_string(cpu_str)

    def set_schedule_strategy(
        self,
        isolate: str = ISOL_AUTO,
        npu_process_cluster_mode: str = NPU_PROCESS_CLUSTER_MODE_ISOLATED,
        enable_npu_topo_affinity: bool = False,
    ) -> None:
        self.isolate = isolate
        self.npu_process_cluster_mode = npu_process_cluster_mode
        self.enable_npu_topo_affinity = enable_npu_topo_affinity
