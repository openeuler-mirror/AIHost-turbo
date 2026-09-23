from __future__ import annotations
from collections import defaultdict

from a_sched.affinity_domain import AffinityDomainManager
from a_sched.task import TaskManager
from a_sched.config import AffinityConfig
from a_sched.strategy.scheduler_base import SchedulerBase
from a_sched.strategy.scheduler_factory import SchedulerFactory


@SchedulerFactory.register("hbs_isolate_cluster")
class HbsIsolateOnCluster(SchedulerBase):
    """
    基于cluster隔离的分层均衡亲和调度策略
    """

    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        super().__init__(config=config, domain=domain, task=task)

        self._numa_to_task_groups: dict[int, list[int]] = defaultdict(list)  # 分配给每个numa的task groups
        self._task_group_to_clusters: dict[int, list[int]] = defaultdict(list)  # 分配给每个task group的clusters
        self._numa_to_background_clusters: dict[int, list[int]] = defaultdict(list)  # 每个numa预留给背景任务的cluster

    def schedule(self) -> None:
        """
        统一调度入口
        """

        # 均衡分发task groups到所有numas
        self._distribute_task_groups_across_numas()

        # 为背景任务分配cpu
        self._assign_cpus_for_background_tasks()

        # 遍历numa，对numa内task groups进行clusters的分配
        for numa_id in self._numa_to_task_groups.keys():
            self._assign_clusters_for_task_groups_in_numa(numa_id=numa_id)

        # 遍历task group，对task group内tasks进行cpus的分配
        for task_group_id in self._task_group_to_clusters.keys():
            self._assign_cpus_for_tasks_in_task_group(task_group_id=task_group_id)

    def _distribute_task_groups_across_numas(self) -> None:
        """
        均分task groups到所有NUMA
        """

        task_groups = self._get_task_groups_to_schedule()
        if len(task_groups) == 0:
            raise RuntimeError(f"no valid task group found to distribute")

        numas = self.domain.get_all_numas_id()
        if len(numas) == 0:
            raise RuntimeError(f"no numa found to distribute")

        numa_to_task_groups = self._balance_distribute_task_groups_to_domains(task_groups=task_groups, domains=numas)

        self._numa_to_task_groups.update(numa_to_task_groups)

    def _assign_clusters_for_task_groups_in_numa(self, numa_id: int) -> None:
        """
        为numa内的task groups分配cluster
        """

        numa = self.domain.get_numa_domain(numa_id=numa_id)
        if numa is None:
            raise RuntimeError(f"numa domain [{numa_id}] not found")

        task_groups = self._numa_to_task_groups.get(numa_id, [])
        if len(task_groups) == 0:
            raise RuntimeError(f"no task group found to assign in numa [{numa_id}]")

        all_clusters = numa.get_all_children_id()
        reserved_clusters = self._numa_to_background_clusters.get(numa_id, [])
        use_clusters = set(all_clusters) - set(reserved_clusters)

        if len(use_clusters) == 0:
            raise RuntimeError(f"no clusters found to assign in numa [{numa_id}], reservd={reserved_clusters}")

        # 所有task groups均分所有clusters
        task_group_to_clusters = self._balance_distribute_task_groups_to_clusters(
            task_groups=task_groups, clusters=sorted(use_clusters)
        )

        self._task_group_to_clusters.update(task_group_to_clusters)

    def _assign_cpus_for_tasks_in_task_group(self, task_group_id: int) -> None:
        """
        task group内为每个task分配cpu
        """

        task_group = self.task.get_group(task_group_id)
        if task_group is None:
            raise RuntimeError(f"task group [{task_group_id}] not found")

        clusters = self._task_group_to_clusters.get(task_group_id, [])
        clusters = self._sort_clusters_by_cpu(clusters=clusters)

        clusters_size = len(clusters)
        if clusters_size == 0:
            raise RuntimeError(f"no clusters found for task group [{task_group_id}]")

        high_prio_tasks = task_group.get_high_prio_tasks()
        min_need_clusters, prefer_need_clusters = self._clusters_needed_by_high_prio_tasks(
            high_prio_tasks=high_prio_tasks
        )

        # 以下场景，不再进一步拆分，group中所有task共享cluster中所有cpu
        # 1） group中没有高优先级任务；
        # 2） 有高优先级任务，但可用cluster数量少于或等于高优先级任务所需clsuter数量，需要保证普通任务能分到cpu
        if min_need_clusters == 0 or clusters_size <= min_need_clusters:
            self._assign_clusters_cpus_to_tasks(clusters=clusters, tasks=task_group.all_tasks)
            # 刷新task group的普通亲和信息
            self._update_task_group_normal_affinity(group=task_group, clusters=clusters)
            return

        # 为高优先级任务分配独享的cpu
        # 如果cluster数量超过高优先级任务需要的最佳数量prefer_need_clusters，优先按照prefer_need_clusters分配
        if clusters_size > prefer_need_clusters:
            isolate_clusters = clusters[-prefer_need_clusters:]
            isolate_used_clusters_size = prefer_need_clusters
        else:
            isolate_clusters = clusters[-min_need_clusters:]
            isolate_used_clusters_size = min_need_clusters
        self._distribute_clusters_cpus_to_tasks(clusters=isolate_clusters, tasks=high_prio_tasks)

        # 刷新task group的隔离亲和信息
        self._update_task_group_isolate_affinity(group=task_group)

        # 普通任务共享剩余cluster的cpu
        normal_clusters = clusters[:-isolate_used_clusters_size]
        self._assign_clusters_cpus_to_tasks(clusters=normal_clusters, tasks=task_group.get_normal_prio_tasks())

        # 刷新task group的普通亲和信息
        self._update_task_group_normal_affinity(group=task_group, clusters=normal_clusters)

    def _assign_cpus_for_background_tasks(self) -> None:
        """
        为背景任务分配clusters
        优先使用空闲numa的cpu，如果没有空闲numa在非隔离numa中预留cpu资源
        """

        normal_numas: set[int] = set()
        for numa in self._numa_to_task_groups.keys():
            normal_numas.add(numa)

        all_numas = set(self.domain.get_all_numas_id())
        unused_numas = all_numas - normal_numas

        self._numa_to_background_clusters = self._do_assign_cpus_for_background_tasks(
            unused_numas=sorted(unused_numas), to_resrve_numas=sorted(normal_numas)
        )
