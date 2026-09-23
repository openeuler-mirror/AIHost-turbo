from __future__ import annotations
from collections import defaultdict

from a_sched.affinity_domain import AffinityDomainManager
from a_sched.task import TaskManager, Task, TaskGroup
from a_sched.config import AffinityConfig
from a_sched.scheduler import Scheduler
import a_sched.utils as utils


class SchedulerBase(Scheduler):
    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        super().__init__(config=config, domain=domain, task=task)

    def _get_task_groups_to_schedule(self) -> list[int]:
        """
        获取待调度的所有task group
        """

        schedule_task_groups: list[int] = []
        for group_id, group in self.task.groups.items():
            if group.all_tasks_num != 0:
                schedule_task_groups.append(group_id)
        return schedule_task_groups

    def _balance_distribute_task_groups_to_domains(
        self, task_groups: list[int], domains: list[int]
    ) -> dict[int, list[int]]:
        """
        将task groups均衡分发到domain（socket / numa)
        """

        task_groups_size = len(task_groups)
        if task_groups_size == 0:
            raise RuntimeError(f"size of task groups is 0")

        domains_size = len(domains)
        if domains_size == 0:
            raise RuntimeError(f"size of domains is 0")

        base = task_groups_size // domains_size
        extra = task_groups_size % domains_size

        domain_to_task_groups: dict[int, list[int]] = defaultdict(list)
        start = 0
        for i, domain_id in enumerate(domains):
            take = base + (1 if i < extra else 0)
            end = start + take
            take_task_groups = task_groups[start:end]
            if not take_task_groups:
                break
            domain_to_task_groups[domain_id].extend(take_task_groups)
            start = end

        return domain_to_task_groups

    def _balance_distribute_task_groups_to_clusters(
        self, task_groups: list[int], clusters: list[int]
    ) -> dict[int, list[int]]:
        """
        为task groups均衡分配clusters
        """

        task_group_to_clusters: dict[int, list[int]] = defaultdict(list)

        task_groups_size = len(task_groups)
        if task_groups_size == 0:
            raise RuntimeError(f"size of task groups is 0")

        clusters_size = len(clusters)
        if clusters_size == 0:
            raise RuntimeError(f"size of clusters is 0")

        # clusters数量少于task groups数量时，不再做细粒度拆分，每个亲和组分享所有cluster
        if clusters_size < task_groups_size:
            print(
                f"[Warning] clusters size ({clusters_size}) < task groups number ({task_groups_size}), "
                f"assign all clusters to each task group"
            )
            for task_group_id in task_groups:
                task_group_to_clusters[task_group_id].extend(clusters)
            return task_group_to_clusters

        base = clusters_size // task_groups_size
        start = 0
        for task_group_id in task_groups:
            end = start + base
            take_clusters = clusters[start:end]
            task_group_to_clusters[task_group_id].extend(take_clusters)
            start = end

        return task_group_to_clusters

    def _distribute_clusters_cpus_to_tasks(self, clusters: list[int], tasks: list[Task]) -> None:
        """
        按照每个task所需的clsuter数量和cpu数量，将clusters中的cpu分配到task
        """

        if len(clusters) == 0:
            raise RuntimeError(f"distribute clusters cpus to tasks fail, invalid clusters")

        sorted_clusters = self._sort_clusters_by_cpu(clusters=clusters)
        all_cpus = self.domain.get_cpus_of_clusters(clusters=clusters)

        start = 0
        for task in tasks:
            task_min_cpu = task.min_cpu

            take_cpus: list[int] = []
            while len(take_cpus) < task_min_cpu:
                end = start + 1
                take_clusters = sorted_clusters[start:end]
                if not take_clusters:
                    break
                take_cpus.extend(self.domain.get_cpus_of_clusters(clusters=take_clusters))
                start = end

            if len(take_cpus) < task_min_cpu:
                print(f"[Warning] no enough clusters to distribute by cluster-level, fallback to cpu-level")
                self._distribute_cpus_to_tasks(cpus=all_cpus, tasks=tasks)
                return

            self._update_task_affinity(task=task, cpus=take_cpus)

    def _distribute_cpus_to_tasks(self, cpus: list[int], tasks: list[Task]) -> None:
        """
        按照每个task所需的cpu数量，将cpus逐task进行分配
        """

        cpus_size = len(cpus)
        if cpus_size == 0:
            raise RuntimeError(f"distribute cpus to tasks fail, invalid cpus, tasks={[t.task_id for t in tasks]}")

        # cpus数量少于tasks需要的数量时，不再进行细粒度分配，所有tasks共享所有cpus
        tasks_need_cpus = self._get_tasks_need_cpus(tasks=tasks)
        if cpus_size < tasks_need_cpus:
            print(
                f"[Warning] no enough cpus when distribute clusters cpus to tasks, "
                f"need {tasks_need_cpus}, actual {cpus_size}, "
                f"cpus={cpus}, tasks={[t.task_id for t in tasks]}"
            )
            self._assign_cpus_to_tasks(cpus=cpus, tasks=tasks)
            return

        start = 0
        for task in tasks:
            end = start + task.min_cpu
            take_cpus = cpus[start:end]
            self._update_task_affinity(task=task, cpus=take_cpus, share=False)
            start = end

    def _assign_clusters_cpus_to_tasks(self, clusters: list[int], tasks: list[Task]) -> None:
        """
        为tasks分配clusters的cpu，所有tasks共享所有clusters的cpu
        """

        cpus = self.domain.get_cpus_of_clusters(clusters=clusters)
        self._assign_cpus_to_tasks(cpus=cpus, tasks=tasks)

    def _assign_cpus_to_tasks(self, cpus: list[int], tasks: list[Task]) -> None:
        """
        为tasks分配cpu，所有tasks共享所有cpu
        """

        if len(cpus) == 0:
            raise RuntimeError(f"assign cpus to tasks fail, invalid cpus, tasks={[t.task_id for t in tasks]}")

        for task in tasks:
            self._update_task_affinity(task=task, cpus=cpus, share=True)

    def _update_task_affinity(self, task: Task, cpus: list[int], share: bool = False) -> None:
        """
        刷新task的亲和分配信息

        share：Ture表示cpus是共享的，task内部不做进一步分配；False表示task内部可以做细粒度分配
        """

        if len(cpus) == 0:
            raise RuntimeError(f"update task affinity fail, invalid cpus, task=[{task.task_id}]")

        task.socket = self.domain.get_sockets_of_cpus(cpus=cpus)
        task.numa = self.domain.get_numas_of_cpus(cpus=cpus)
        task.cluster = self.domain.get_clusters_of_cpus(cpus=cpus)
        task.set_cpu(cpus) if share else task.assign_cpu(cpus)

    def _update_task_group_normal_affinity(self, group: TaskGroup, clusters: list[int]) -> None:
        """
        刷新task group的亲和分配信息
        """

        if len(clusters) == 0:
            raise RuntimeError(f"update task group affinity fail, invalid clusters, group=[{group.group_id}]")

        group.socket = self.domain.get_sockets_of_clusters(clusters)
        group.numa = self.domain.get_numas_of_clusters(clusters)
        group.cluster = sorted(clusters)
        group.cpus.set_list(self.domain.get_cpus_of_clusters(clusters))

    def _update_task_group_isolate_affinity(self, group: TaskGroup) -> None:
        """
        更新高优先级任务的隔离亲和信息到所在的task group
        """

        for task in group.get_high_prio_tasks():
            group.isolate_numa.extend(task.numa)
            group.isolate_cluster.extend(task.cluster)
            group.isolate_cpus.set_list(task.cpus.to_list())
            group.isolate_numa = sorted(set(group.isolate_numa))
            group.isolate_cluster = sorted(set(group.isolate_cluster))

    def _get_tasks_need_cpus(self, tasks: list[Task]) -> int:
        """
        获取taks需要的最小cpu数量
        """

        return sum(task.min_cpu for task in tasks)

    def _sort_clusters_by_cpu(self, clusters: list[int]) -> list[int]:
        """
        将clusters按照各自实际cpu数量排序，从多到少
        """

        def get_cpu_count(cluster_id: int) -> int:
            cluster = self.domain.get_cluster_domain(cluster_id=cluster_id)
            return cluster.cpus.count() if cluster is not None else 0

        return sorted(clusters, key=get_cpu_count, reverse=True)

    def _clusters_needed_by_high_prio_tasks(self, high_prio_tasks: list[Task]):
        """
        获取高优先级task需要的cluster数量

        Return: (最小cluster数量，最佳cluster数量)
        """

        cluster_cpu_size = self.domain.get_cluster_cpu_size()
        if cluster_cpu_size == 0:
            raise RuntimeError("max size of cluster cpu is 0")

        need_cpus = self._get_tasks_need_cpus(tasks=high_prio_tasks)
        min_clusters = (need_cpus + cluster_cpu_size - 1) // cluster_cpu_size

        prefer_clusters = 0
        for task in high_prio_tasks:
            task_need_clusters = (task.min_cpu + cluster_cpu_size - 1) // cluster_cpu_size
            prefer_clusters += task_need_clusters

        return min_clusters, prefer_clusters

    def _reserve_clusters_of_numas_for_background_tasks(self, numas: list[int]) -> dict[int, list[int]]:
        """
        指定numa中为背景任务预留cluster资源

        Return：dict[numa_id: int, reserved_clusters: list[int]]
        """

        numa_to_reserved_clusters: dict[int, list[int]] = defaultdict(list)

        for numa_id in numas:
            numa = self.domain.get_numa_domain(numa_id=numa_id)
            if numa is None:
                print(f"[Error] numa domain {numa_id} not found")
                continue

            clusters = numa.get_all_children_id()
            clusters_num = len(clusters)
            if clusters_num < 2:
                print(f"no enough clusters in numa [{numa_id}] for background tasks, actual={clusters_num} ")
                continue

            reserve_count = max(1, int(clusters_num * 0.2))
            numa_to_reserved_clusters[numa_id] = clusters[:reserve_count]
            print(
                f"numa[{numa_id}] reserve {reserve_count} cluster(s) for background tasks, "
                f"{clusters_num - reserve_count} cluster(s) left for normal tasks"
            )

        return numa_to_reserved_clusters

    def _do_assign_cpus_for_background_tasks(
        self, unused_numas: list[int], to_resrve_numas: list[int]
    ) -> dict[int, list[int]]:
        """
        为背景任务分配clusters
        优先使用空闲numa的cpu，如果没有空闲numa在非隔离numa中预留cpu资源

        Return：dict[numa_id: int, reserved_clusters: list[int]]
        """

        # 优先使用空闲numa的cpu
        if unused_numas:
            cpus = self.domain.get_cpus_of_numas(numas=unused_numas)
            print(
                f"using unused NUMA CPUs for background tasks: cpus={utils.compress_continuous(cpus)}, "
                f"numa={sorted(unused_numas)}"
            )
            self.task.background_tasks_cpus = cpus
            return {}

        # 如果没有空闲numa，在普通numa中预留cpu资源
        numa_to_background_clusters = self._reserve_clusters_of_numas_for_background_tasks(
            numas=sorted(to_resrve_numas)
        )
        reserved_cpus = []
        for _, clusters in numa_to_background_clusters.items():
            reserved_cpus.extend(self.domain.get_cpus_of_clusters(clusters=clusters))
        if reserved_cpus:
            print(
                f"using reserved clusters from each NUMA: cpus={utils.compress_continuous(reserved_cpus)}, "
                f"(numa={to_resrve_numas})"
            )
            self.task.background_tasks_cpus = reserved_cpus
            return numa_to_background_clusters

        # 未找到可以分配的CPU
        print("no target CPUs found for background tasks")
        return {}
