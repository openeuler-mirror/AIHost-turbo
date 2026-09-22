from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto

from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils


class PriorityLevel(Enum):
    """任务优先级"""

    NORMAL = auto()
    HIGH = auto()


class TaskType(Enum):
    """任务类型"""

    THREAD = auto()  # 线程
    PROCESS = auto()  # 进程
    NPU_A3 = auto()  # NPU_A3
    NPU_A5 = auto()  # NPU_A5


class Task(ABC):
    def __init__(self, task_type: TaskType, group_id: int, task_id: int, name: str | None = None):
        # 基本信息
        self.group_id: int = group_id  # task所属组的标识
        self.task_id: int = task_id  # task标识，相同task类型内task标识必须唯一
        self.name: str | None = name  # task名称
        self.task_type: TaskType = task_type  # task类型
        self.priority: PriorityLevel = PriorityLevel.NORMAL  # task优先级，默认优先级为Normal
        self.bind_npu: int | None = None  # task绑定的npu，默认为空

        # 调度管理
        self.cpus = utils.CPUMask()  # 分配到的cpu
        self.socket: list[int] = []  # 调度到的socket
        self.numa: list[int] = []  # 调度到的numa
        self.cluster: list[int] = []  # 调度到的cluster

        self._type_names = {
            TaskType.THREAD: "THREAD",
            TaskType.PROCESS: "PROCESS",
            TaskType.NPU_A3: "NPU_A3",
            TaskType.NPU_A5: "NPU_A5"
        }

        self._priority_names = {
            PriorityLevel.NORMAL: "NORMAL",
            PriorityLevel.HIGH: "HIGH"
        }

    @property
    def min_cpu(self) -> int:
        return 1

    def set_cpu(self, cpus: list[int]) -> None:
        self.cpus = utils.CPUMask().from_list(cpus)

    @abstractmethod
    def assign_cpu(self, cpus: list[int]) -> None:
        pass

    @abstractmethod
    def bind_cpu(self) -> None:
        pass

    @abstractmethod
    def print_actual_affinity(self, domain: AffinityDomainManager) -> None:
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
