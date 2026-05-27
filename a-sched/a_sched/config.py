import a_sched.utils as utils


class AffinityConfig:

    def __init__(self) -> None:
        self.exclude_cpus: list[int] = []

    def set_exclude_cpu(self, cpu_str: str) -> None:
        self.exclude_cpus = utils.parse_cpu_affinity_string(cpu_str)
