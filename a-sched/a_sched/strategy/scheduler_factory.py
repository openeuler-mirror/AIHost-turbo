from a_sched.affinity_domain import AffinityDomainManager
from a_sched.task import TaskManager
from a_sched.config import AffinityConfig
import a_sched.config as cfg
import a_sched.utils as utils

HBS_ISOLATE_CLUSTER = "hbs_isolate_cluster"
HBS_ISOLATE_NUMA = "hbs_isolate_numa"


def decide_scheduler(config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager) -> str:
    """
    决策调度器类型
    """

    if config.isolate == cfg.ISOL_CLUSTER:
        return decide_on_isol_cluster(config=config, domain=domain, task=task)
    elif config.isolate == cfg.ISOL_NUMA:
        return decide_on_isol_numa(config=config, domain=domain, task=task)
    elif config.isolate == cfg.ISOL_AUTO:
        return decide_on_isol_auto(config=config, domain=domain, task=task)
    else:
        raise ValueError(f"undefined isolate strategy: {config.isolate}")


def decide_on_isol_cluster(config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager) -> str:
    """
    isolate策略为“cluster”级别隔离时，决策调度器类型
    """

    return HBS_ISOLATE_CLUSTER


def decide_on_isol_numa(config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager) -> str:
    """
    isolate策略为“numa”级别隔离时，决策调度器类型
    """

    if len(domain.socket_domains) == 0:
        raise RuntimeError(f"no valid socket to schedule")

    # 如果socket中numa数量小于2个，无法进行numa隔离，调整为cluster隔离
    for socket in domain.socket_domains:
        if socket.get_children_num() < 2:
            return decide_on_isol_cluster(config=config, domain=domain, task=task)

    # 使用hbs_isolate_numa调度器
    return HBS_ISOLATE_NUMA


def decide_on_isol_auto(config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager) -> str:
    """
    isolate策略为“auto”时，决策调度器类型
    """

    device_type = utils.get_ascend_device_type()
    if device_type == utils.AscendDeviceType.A3:
        # A3默认使用NUMA级别隔离
        return decide_on_isol_numa(config=config, domain=domain, task=task)
    elif device_type == utils.AscendDeviceType.A5:
        # A5默认使用cluster级别隔离
        return decide_on_isol_cluster(config=config, domain=domain, task=task)
    else:
        raise RuntimeError(f"unsupported ascend device type {device_type}")


class SchedulerFactory:
    _scheduler_map: dict[str, type] = {}

    @classmethod
    def register(cls, scheduler_type: str):
        def wrapper(scheduler_cls):
            cls._scheduler_map[scheduler_type] = scheduler_cls
            return scheduler_cls

        return wrapper

    @classmethod
    def create(cls, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        scheduler_type = decide_scheduler(config=config, domain=domain, task=task)
        print(f"use scheduler: {scheduler_type}")
        scheduler_cls = cls._scheduler_map.get(scheduler_type)
        if not scheduler_cls:
            raise ValueError(f"unknown scheduler: {scheduler_type}")
        return scheduler_cls(config, domain, task)
