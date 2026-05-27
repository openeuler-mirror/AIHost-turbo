from abc import ABC, abstractmethod

from a_sched.affinity_domain import AffinityDomainManager
from a_sched.config import AffinityConfig
from a_sched.task import TaskManager


class Scheduler(ABC):
    """调度器抽象基类"""

    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        self.config = config
        self.domain = domain
        self.task = task

    @abstractmethod
    def schedule(self) -> bool:
        return True
