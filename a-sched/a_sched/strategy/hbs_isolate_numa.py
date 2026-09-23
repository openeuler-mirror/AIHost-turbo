from __future__ import annotations
from collections import defaultdict

from a_sched.affinity_domain import AffinityDomainManager, SocketDomain, NumaDomain
from a_sched.task import TaskManager, TaskGroup, Task
from a_sched.config import AffinityConfig
from a_sched.strategy.scheduler_base import SchedulerBase
from a_sched.strategy.scheduler_factory import SchedulerFactory


@SchedulerFactory.register("hbs_isolate_numa")
class HbsIsolateOnNuma(SchedulerBase):
    """
    基于numa隔离的分层均衡亲和调度策略
    """

    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        super().__init__(config=config, domain=domain, task=task)

        self._socket_to_task_groups: dict[int, list[int]] = defaultdict(list)  # 分配给每个socket的亲和组
        self._socket_to_normal_numas: dict[int, list[int]] = defaultdict(list)  # socket中的非隔离numa
        self._socket_to_isolate_numa: dict[int, int] = {}  # socket中用于隔离的numa
        self._socket_to_high_prio_tasks: dict[int, list[Task]] = defaultdict(list)  # 每个socket中待调度的高优先级task
        self._numa_to_task_groups: dict[int, list[int]] = defaultdict(list)  # 分配给各numa的亲和组
        self._task_group_to_clusters: dict[int, list[int]] = defaultdict(list)  # 分配给每个亲和组的clusters
        self._numa_to_background_clusters: dict[int, list[int]] = defaultdict(list)  # 每个numa预留给背景任务的cluster

    def schedule(self) -> None:
        """
        统一调度入口
        """

        # 均衡分发task groups到所有sockets
        self._distribute_task_groups_across_sockets()

        # 遍历socket，对socket内的task groups进行调度处理
        # 1）选择隔离numa，高优先级任务在隔离numa内分配资源
        # 2）task groups均衡分发到非隔离numa
        for socket_id in self._socket_to_task_groups.keys():
            self._schedule_task_groups_of_socket(socket_id)

        # 为背景任务分配cpu
        self._assign_cpus_for_background_tasks()

        # 遍历numa，对numa内task groups进行clusters的分配
        for numa_id in self._numa_to_task_groups.keys():
            self._assign_clusters_for_task_groups_in_numa(numa_id)

        # 遍历task group，对task group内tasks进行cpus的分配
        for task_group_id in self._task_group_to_clusters.keys():
            self._assign_cpus_for_normal_tasks_in_task_group(task_group_id)

    def _distribute_task_groups_across_sockets(self) -> None:
        """
        均分所有task groups到所有sockets
        """

        task_groups = self._get_task_groups_to_schedule()
        if len(task_groups) == 0:
            raise RuntimeError(f"no valid task group found to distribute")

        sockets = self.domain.get_all_sockets_id()
        if len(sockets) == 0:
            raise RuntimeError(f"no socket found to distribute")

        self._socket_to_task_groups = self._balance_distribute_task_groups_to_domains(
            task_groups=task_groups, domains=sockets
        )

    def _schedule_task_groups_of_socket(self, socket_id: int) -> None:
        """
        对单个socket的task groups进行调度
        """

        socket = self.domain.get_socket_domain(socket_id)
        if socket is None:
            raise RuntimeError(f"socket domain [{socket_id}] not found")

        # numa隔离预处理
        self._prepare_numa_isolate_for_socket(socket)

        # 隔离numa内对高优先级任务进行分配
        self._assign_clusters_for_high_prio_tasks_in_isolate_numa(socket_id=socket_id)

        # 刷新高优先级task的亲和信息到task group
        self._update_isolate_affinity_of_socket_task_group(socket_id=socket_id)

        # 非隔离numa对task group进行分配
        self._distribute_socket_task_groups_to_normal_numas(socket_id=socket_id)

    def _prepare_numa_isolate_for_socket(self, socket: SocketDomain) -> None:
        """
        numa隔离预处理，决策是否需要进行numa隔离
        """

        socket_numas = sorted(socket.get_all_children_id())
        # socket内numa数量为1场景，在外层做判断后直接走cluster隔离处理，在此处要求nuam数量大于1
        if len(socket_numas) <= 1:
            raise RuntimeError(f"numas of socket [{socket.domain_id}] need > 1")

        # 获取高优先级任务
        for group_id in self._socket_to_task_groups[socket.domain_id]:
            group_high_prio_tasks = self.task.get_high_prio_tasks_of_group(group_id)
            self._socket_to_high_prio_tasks[socket.domain_id].extend(group_high_prio_tasks)

        # 没有高优先级任务，不需要进行numa隔离
        if len(self._socket_to_high_prio_tasks[socket.domain_id]) == 0:
            self._socket_to_normal_numas[socket.domain_id] = socket_numas
            return

        # 使用socket中最后一个numa作为隔离域numa，其他的numa作为非隔离域numa
        self._socket_to_normal_numas[socket.domain_id].extend(socket_numas[:-1])
        self._socket_to_isolate_numa[socket.domain_id] = socket_numas[-1]

    def _assign_clusters_for_high_prio_tasks_in_isolate_numa(self, socket_id: int) -> None:
        """
        为高优先级任务在隔离numa内分配cluster
        """

        socket_high_prio_tasks = self._socket_to_high_prio_tasks[socket_id]
        # 没有高优先级任务，直接返回
        if not socket_high_prio_tasks:
            return

        isolate_numa_id = self._socket_to_isolate_numa.get(socket_id)
        if isolate_numa_id is None:
            raise RuntimeError(f"invalid isolate numa")

        isolate_numa = self.domain.get_numa_domain(isolate_numa_id)
        if isolate_numa is None:
            raise RuntimeError(f"isolate numa [{isolate_numa_id}] not found")

        clusters = isolate_numa.get_all_children_id()
        self._distribute_clusters_cpus_to_tasks(clusters=clusters, tasks=socket_high_prio_tasks)

    def _distribute_socket_task_groups_to_normal_numas(self, socket_id: int) -> None:
        """
        均分socket的task groups到非隔离numa
        """

        task_groups = self._socket_to_task_groups[socket_id]
        if len(task_groups) == 0:
            raise RuntimeError(f"no task group to distribute in socket [{socket_id}]")

        normal_numas = self._socket_to_normal_numas[socket_id]
        if len(normal_numas) == 0:
            raise RuntimeError(f"no normal numa to distribute in socket {socket_id}")

        numa_to_task_groups = self._balance_distribute_task_groups_to_domains(
            task_groups=task_groups, domains=normal_numas
        )

        self._numa_to_task_groups.update(numa_to_task_groups)

    def _assign_clusters_for_task_groups_in_numa(self, numa_id: int) -> None:
        """
        为numa内的task groups分配cluster
        """

        numa = self.domain.get_numa_domain(numa_id)
        if numa is None:
            raise RuntimeError(f"numa domain [{numa_id}] not found")

        task_groups = self._numa_to_task_groups[numa_id]
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

    def _assign_cpus_for_normal_tasks_in_task_group(self, task_group_id: int) -> None:
        """
        task group内为每个normal task分配cpu
        """

        task_group = self.task.get_group(task_group_id)
        if task_group is None:
            raise RuntimeError(f"task group [{task_group_id}] not found")

        clusters = self._task_group_to_clusters.get(task_group_id, [])
        if len(clusters) == 0:
            raise RuntimeError(f"no clusters found for task group [{task_group_id}]")

        # 所有普通task共享所有cluster的cpu
        self._assign_clusters_cpus_to_tasks(clusters=clusters, tasks=task_group.get_normal_prio_tasks())

        # 刷新task group的亲和信息
        self._update_task_group_normal_affinity(group=task_group, clusters=clusters)

    def _update_isolate_affinity_of_socket_task_group(self, socket_id: int) -> None:
        """
        更新socket中task group的隔离亲和信息
        """

        task_groups = self._socket_to_task_groups[socket_id]
        for group_id in task_groups:
            group = self.task.get_group(group_id=group_id)
            if group is None:
                raise RuntimeError(f"task group [{group_id}] not found")
            self._update_task_group_isolate_affinity(group=group)

    def _assign_cpus_for_background_tasks(self) -> None:
        """
        为背景任务分配clusters
        优先使用空闲numa的cpu，如果没有空闲numa在非隔离numa中预留cpu资源
        """

        normal_numas: set[int] = set()
        for numa in self._numa_to_task_groups.keys():
            normal_numas.add(numa)

        isolate_numas: set[int] = set()
        for _, isol_numa in self._socket_to_isolate_numa.items():
            isolate_numas.add(isol_numa)

        all_numas = set(self.domain.get_all_numas_id())
        unused_numas = all_numas - normal_numas - isolate_numas

        self._numa_to_background_clusters = self._do_assign_cpus_for_background_tasks(
            unused_numas=sorted(unused_numas), to_resrve_numas=sorted(normal_numas)
        )
