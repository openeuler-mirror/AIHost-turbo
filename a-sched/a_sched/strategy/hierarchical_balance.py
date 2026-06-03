from __future__ import annotations
from collections import defaultdict

from a_sched.affinity_domain import AffinityDomainManager, SocketDomain, NumaDomain
from a_sched.task import TaskManager, TaskGroup, Task
from a_sched.config import AffinityConfig
from a_sched.scheduler import Scheduler
import a_sched.utils as utils


class HierarchicalBalanceScheduler(Scheduler):
    """分层均衡亲和调度器"""

    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        super().__init__(config=config, domain=domain, task=task)

        # 待调度的亲和组，非空，至少有一个task
        self._schedule_task_groups: list[int] = []
        # 可被调度的socket，需要剔除部分资源不足的socket
        self._schedule_sockets: list[int] = []

        # 调度到各socket的亲和组
        self._socket_to_task_groups: dict[int, list[int]] = defaultdict(list)
        # 每个socket中待调度的高优先级task
        self._socket_to_high_prio_tasks: dict[int, list[Task]] = defaultdict(list)
        # socket除隔离域numa外的其他正常numa
        self._socket_to_normal_numas: dict[int, list[int]] = defaultdict(list)
        # socket用于隔离域的numa
        self._socket_to_isolate_numa: dict[int, int] = {}
        # 单NUMA场景下，高优先级任务占用的CPU列表
        self._socket_to_isolate_cpus: dict[int, list[int]] = defaultdict(list)

        # 调度到各numa的亲和组
        self._numa_to_task_groups: dict[int, list[int]] = defaultdict(list)
        # 每个numa预留给背景任务的cluster
        self._numa_to_background_clusters: dict[int, list[int]] = defaultdict(list)
        # 分配给每个亲和组的cluster
        self._task_group_to_clusters: dict[int, list[int]] = defaultdict(list)

    def schedule(self) -> bool:
        # 获取待调度的非空亲和组
        self._get_task_groups_to_schedule()
        if len(self._schedule_task_groups) == 0:
            print("[Error] no valid task group found to schedule.")
            return False

        # 获取可参与调度的socket
        self._get_schedule_sockets()
        if len(self._schedule_sockets) == 0:
            print("[Error] no valid socket found to schedule.")
            return False

        # socket层对等均分
        if not self._schedule_task_groups_to_sockets():
            return False

        # 每个socket内，对task group进行numa层的对等均分
        for socket_id in self._socket_to_task_groups:
            if not self._schedule_task_groups_of_socket(socket_id):
                return False

        # 每个numa内，对task group进行cluster的对等均分
        for numa_id in self._numa_to_task_groups:
            if not self._schedule_numa_task_groups_to_clusters(numa_id):
                return False

        # 为每个task group刷新affinity
        for _, group in self.task.groups.items():
            if not self._update_task_group_affinity(group):
                return False

        # 每个task group内，对每个task（进程/线程）进行分配
        for _, group in self.task.groups.items():
            if not self._schedule_tasks_affinity_of_group(group):
                return False

        # 刷新高优先级task的亲和信息给所在的group
        for _, group in self.task.groups.items():
            self._update_task_group_with_high_prio_tasks(group)

        # 背景任务分配cpu
        self._alloc_cpus_for_background_tasks()

        return True

    def _get_task_groups_to_schedule(self) -> None:
        groups_to_schedule: list[int] = []
        for group_id, group in self.task.groups.items():
            if group.get_all_tasks_num() != 0:
                groups_to_schedule.append(group_id)
        self._schedule_task_groups = groups_to_schedule

    def _get_schedule_sockets(self) -> None:
        schedule_sockets: list[int] = []
        for socket in self.domain.socket_domains:
            schedule_sockets.append(socket.domain_id)
        self._schedule_sockets = schedule_sockets

    def _schedule_task_groups_to_sockets(self) -> bool:
        total_task_group_num = len(self._schedule_task_groups)
        total_socket_num = len(self._schedule_sockets)
        if total_socket_num == 0:
            print("no valid socket found.")
            return False

        base = total_task_group_num // total_socket_num
        extra = total_task_group_num % total_socket_num

        start = 0
        for i, socket_id in enumerate(self._schedule_sockets):
            take = base + (1 if i < extra else 0)
            end = start + take
            take_task_groups = self._schedule_task_groups[start:end]
            if not take_task_groups:
                break
            self._socket_to_task_groups[socket_id].extend(take_task_groups)
            start = end

        return True

    def _schedule_task_groups_of_socket(self, socket_id: int) -> bool:
        socket = self.domain.get_socket_domain(socket_id)
        if socket is None:
            print(f"socket domain [{socket_id}] not found.")
            return False

        # numa隔离预处理
        if not self._schedule_isolate_on_numa_of_socket(socket):
            return False

        # 高优先级任务分配, 基于是否存在 isolate_numa 自动分流
        if not self._dispatch_high_prio_tasks_by_topology(socket):
            return False

        # numa非隔离域内对task group进行调度
        socket_normal_numas = self._socket_to_normal_numas[socket_id]
        if not self._schedule_socket_task_groups_to_numas(socket=socket, numas=socket_normal_numas):
            return False

        return True

    def _schedule_isolate_on_numa_of_socket(self, socket: SocketDomain) -> bool:
        socket_numas = sorted(socket.get_all_children_id())
        socket_numa_num = len(socket_numas)
        if socket_numa_num == 0:
            print(f"no numa with online CPU found in socket [{socket.domain_id}]")
            return False

        socket_high_prio_tasks: list[Task] = []
        socket_task_groups = self._socket_to_task_groups[socket.domain_id]
        for group_id in socket_task_groups:
            group_high_prio_tasks = self.task.get_high_prio_tasks_of_group(group_id)
            socket_high_prio_tasks.extend(group_high_prio_tasks)

        # 没有高优先级任务，不需要进行numa隔离
        if len(socket_high_prio_tasks) == 0:
            self._socket_to_normal_numas[socket.domain_id] = socket_numas
            return True

        # 存在高优先级任务，且numa数量小于2时，没有多余的numa可以隔离，不进行numa隔离，改为在cpu层面进行隔离
        if socket_numa_num < 2:
            print(f"[Warning] no enough numa for isolate in socket [{socket.domain_id}].")
            self._socket_to_high_prio_tasks[socket.domain_id] = socket_high_prio_tasks
            self._socket_to_normal_numas[socket.domain_id] = socket_numas
            return True

        self._socket_to_high_prio_tasks[socket.domain_id] = socket_high_prio_tasks

        # 使用socket中最后一个numa作为隔离域numa，其他的numa作为非隔离域numa
        self._socket_to_normal_numas[socket.domain_id].extend(socket_numas[:-1])
        self._socket_to_isolate_numa[socket.domain_id] = socket_numas[-1]

        return True

    def _dispatch_high_prio_tasks_by_topology(self, socket: SocketDomain) -> bool:
        socket_high_prio_tasks = self._socket_to_high_prio_tasks.get(socket.domain_id, [])
        if not socket_high_prio_tasks:
            return True

        isolate_numa_id = self._socket_to_isolate_numa.get(socket.domain_id)

        if isolate_numa_id is not None:
            # 有预留的隔离numa → numa 级别隔离
            return self._schedule_socket_high_prio_tasks_to_isolate_numa(socket)
        else:
            # 1 socket 1 numa  → CPU 级别隔离
            return self._schedule_high_prio_tasks_to_isolate_cpu(socket, socket_high_prio_tasks)

    def _schedule_socket_high_prio_tasks_to_isolate_numa(self, socket: SocketDomain) -> bool:
        socket_high_prio_tasks = self._socket_to_high_prio_tasks[socket.domain_id]
        task_num = len(socket_high_prio_tasks)
        if task_num == 0:
            return True

        isolate_numa_id = self._socket_to_isolate_numa.get(socket.domain_id)
        if isolate_numa_id is None:
            return True

        isolate_numa = self.domain.get_numa_domain(isolate_numa_id)
        if isolate_numa is None:
            print(f"isolate numa domain [{isolate_numa_id}] not found.")
            return False

        clusters = isolate_numa.get_all_children_id()
        cluster_num = len(clusters)

        task_needed_clusters = 0
        for task in socket_high_prio_tasks:
            task_needed_clusters += task.min_cluster

        if len(clusters) == 0:
            print(f"isolate numa [{isolate_numa_id}] has no available cluster")
            return False

        # cluster数量大于等于task需要的cluster数量时，按照task的实际cluster需求分配
        # 需验证每个task分配到的cluster是否有足够的CPU
        if cluster_num >= task_needed_clusters:
            self._alloc_high_prio_tasks_by_cluster(
                socket=socket,
                isolate_numa=isolate_numa,
                clusters=clusters,
                socket_high_prio_tasks=socket_high_prio_tasks,
            )
            return True

        print(
            f"[Warning] no enough clusters in isolate numa [{isolate_numa_id}], "
            f"need {task_needed_clusters}, actual {cluster_num}."
        )

        self._alloc_high_prio_tasks_by_cpu(
            socket=socket, isolate_numa=isolate_numa, socket_high_prio_tasks=socket_high_prio_tasks
        )
        return True

    def _alloc_high_prio_tasks_by_cluster(
        self, socket: SocketDomain, isolate_numa: NumaDomain, clusters: list[int], socket_high_prio_tasks: list[Task]
    ) -> None:
        start = 0
        cluster_count = len(clusters)

        for task in socket_high_prio_tasks:
            if start >= cluster_count:
                print(
                    f"[Warning] no more clusters for task(id={task.task_id}, name={task.name}), "
                    f"fallback to CPU-level allocation."
                )
                self._alloc_high_prio_tasks_by_cpu(
                    socket=socket, isolate_numa=isolate_numa, socket_high_prio_tasks=socket_high_prio_tasks
                )
                return

            end = min(start + task.min_cluster, cluster_count)
            take_cluster = clusters[start:end]
            cpu_list = self._get_cpu_list_of_clusters(take_cluster)

            while len(cpu_list) < task.min_cpu and end < cluster_count:
                end += 1
                take_cluster = clusters[start:end]
                cpu_list = self._get_cpu_list_of_clusters(take_cluster)

            if len(cpu_list) < task.min_cpu:
                print(
                    f"[Warning] clusters {take_cluster} only have {len(cpu_list)} CPUs, "
                    f"task(id={task.task_id}, name={task.name}) needs {task.min_cpu} CPUs, "
                    f"fallback to CPU-level allocation."
                )
                self._alloc_high_prio_tasks_by_cpu(
                    socket=socket, isolate_numa=isolate_numa, socket_high_prio_tasks=socket_high_prio_tasks
                )
                return

            self._update_task_affinity(task, [socket.domain_id], [isolate_numa.domain_id], take_cluster, cpu_list)
            task.do_isolate = True
            start = end

    def _alloc_high_prio_tasks_by_cpu(
        self, socket: SocketDomain, isolate_numa: NumaDomain, socket_high_prio_tasks: list[Task]
    ) -> None:
        task_needed_cpus = sum(task.min_cpu for task in socket_high_prio_tasks)
        cpu_list = isolate_numa.cpus.to_list()
        cpu_num = len(cpu_list)

        if cpu_num >= task_needed_cpus:
            start = 0
            for task in socket_high_prio_tasks:
                end = start + task.min_cpu
                take_core = cpu_list[start:end]
                cluster_ids = self.domain.get_clusters_of_cpus(take_core)
                self._update_task_affinity(task, [socket.domain_id], [isolate_numa.domain_id], cluster_ids, take_core)
                task.do_isolate = True
                start = end
            return

        print(
            f"[Warning] no enough cpus in isolate numa [{isolate_numa.domain_id}], "
            f"need {task_needed_cpus}, actual {cpu_num}."
        )

        clusters = isolate_numa.get_all_children_id()
        for task in socket_high_prio_tasks:
            self._update_task_affinity(task, [socket.domain_id], [isolate_numa.domain_id], clusters, cpu_list)
            task.do_isolate = True

    def _schedule_high_prio_tasks_to_isolate_cpu(
        self, socket: SocketDomain, socket_high_prio_tasks: list[Task]
    ) -> bool:
        """单NUMA场景下,为高优先级任务分配独立CPU"""
        socket_numas = self._socket_to_normal_numas[socket.domain_id]
        if not socket_numas:
            print(f"[Error] no numa found for socket [{socket.domain_id}]")
            return False

        all_clusters = []
        for numa_id in socket_numas:
            numa = self.domain.get_numa_domain(numa_id)
            if numa is not None:
                all_clusters.extend(numa.get_all_children_id())
        all_clusters = sorted(all_clusters)
        cluster_num = len(all_clusters)

        if cluster_num == 0:
            print(f"[Error] no clusters found in socket [{socket.domain_id}]")
            return False

        # 计算高优先级任务需要的CPU数量（每个任务1个CPU）
        task_needed_cpus = 0
        for task in socket_high_prio_tasks:
            task_needed_cpus += task.min_cpu

        # 按CPU数量均分：高优先级任务占用前N个CPU，剩余给普通任务
        cpu_list = []
        for cluster_id in all_clusters:
            cluster_domain = self.domain.get_cluster_domain(cluster_id)
            if cluster_domain:
                cpu_list.extend(cluster_domain.cpus.to_list())
        cpu_list = sorted(cpu_list)

        if len(cpu_list) < task_needed_cpus:
            print(f"[Warning] not enough CPUs for high priority tasks: need {task_needed_cpus}, actual {len(cpu_list)}")
            # 如果CPU不够，所有高优先级任务共享所有CPU
            for task in socket_high_prio_tasks:
                self._update_task_affinity(task, [socket.domain_id], socket_numas, all_clusters, cpu_list)
                task.do_isolate = True
            return True

        start = 0
        for task in socket_high_prio_tasks:
            end = start + task.min_cpu
            take_cpu = cpu_list[start:end]
            cluster_ids = self.domain.get_clusters_of_cpus(take_cpu)
            self._update_task_affinity(task, [socket.domain_id], socket_numas, cluster_ids, take_cpu)
            task.do_isolate = True
            start = end

        self._socket_to_isolate_cpus[socket.domain_id] = cpu_list[:task_needed_cpus]
        return True

    def _schedule_socket_task_groups_to_numas(self, socket: SocketDomain, numas: list[int]) -> bool:
        socket_task_groups = self._socket_to_task_groups[socket.domain_id]
        task_group_num = len(socket_task_groups)

        numa_num = len(numas)
        if numa_num == 0:
            print(f"no numa to schedule for socket{socket.domain_id}")
            return False

        base = task_group_num // numa_num
        extra = task_group_num % numa_num

        start = 0
        for i, numa in enumerate(numas):
            take = base + (1 if i < extra else 0)
            end = start + take
            take_task_groups = socket_task_groups[start:end]
            if not take_task_groups:
                break
            self._numa_to_task_groups[numa].extend(take_task_groups)
            start = end

        return True

    def _schedule_numa_task_groups_to_clusters(self, numa_id: int) -> bool:
        numa_domain = self.domain.get_numa_domain(numa_id)
        if numa_domain is None:
            print(f"numa domain [{numa_id}] not found")
            return False

        cluster_domains = numa_domain.get_all_children_id()

        current_socket_id = None
        for socket_id, normal_numas in self._socket_to_normal_numas.items():
            if numa_id in normal_numas:
                current_socket_id = socket_id
                break

        if current_socket_id is not None:
            isolate_cpus = self._socket_to_isolate_cpus.get(current_socket_id, [])
            if isolate_cpus:
                filtered_clusters = []
                for cluster_id in cluster_domains:
                    cluster_domain = self.domain.get_cluster_domain(cluster_id)
                    if cluster_domain is None:
                        continue
                    cluster_cpus = set(cluster_domain.cpus.to_list())
                    if not cluster_cpus.intersection(isolate_cpus):
                        filtered_clusters.append(cluster_id)
                if filtered_clusters:
                    cluster_domains = filtered_clusters

        cluster_num = len(cluster_domains)

        numa_task_groups = self._numa_to_task_groups[numa_id]
        task_group_num = len(numa_task_groups)
        if task_group_num == 0:
            return True

        if cluster_num == 0:
            print(f"numa[{numa_id}] has no available clusters, skip.")
            return False

        reserve_count = max(1, int(cluster_num * 0.2))
        self._numa_to_background_clusters[numa_id] = cluster_domains[:reserve_count]
        cluster_domains = cluster_domains[reserve_count:]
        cluster_num = len(cluster_domains)
        print(
            f"numa[{numa_id}] reserve {reserve_count} cluster(s) for background tasks, "
            f"{cluster_num} cluster(s) left for normal tasks."
        )

        if cluster_num < task_group_num:
            # cluster数量少于group数量时，不再做细粒度拆分，每个亲和组分享该numa下所有cluster
            print(f"numa [{numa_id}]: cluster number ({cluster_num})  < task group number ({task_group_num})")
            for _, task_group_id in enumerate(numa_task_groups):
                self._task_group_to_clusters[task_group_id].extend(cluster_domains)
            return True

        base = cluster_num // task_group_num
        start = 0
        for _, task_group_id in enumerate(numa_task_groups):
            end = start + base
            take_clusters = cluster_domains[start:end]
            if not take_clusters:
                break
            self._task_group_to_clusters[task_group_id].extend(take_clusters)
            start = end

        return True

    def _update_task_group_affinity(self, group: TaskGroup) -> bool:
        # update task group clusters
        task_group_clusters = self._task_group_to_clusters.get(group.group_id)
        if task_group_clusters is None:
            print(f"no clusters for task group [{group.group_id}]")
            return False
        group.cluster = sorted(task_group_clusters)
        if self.config.enable_cpuset:
            group.cluster = group.cluster[:len(group.cluster) // 2]

        # update task group cpus
        for cluster_id in group.cluster:
            cluster_domain = self.domain.get_cluster_domain(cluster_id)
            if cluster_domain is None:
                print(f"cluster domain [{cluster_id}] not found")
                return False
            group.cpus.set_list(cluster_domain.cpus.to_list())

        # update task group numas
        task_group_numas: list[int] = []
        for numa_id, numa_task_groups in self._numa_to_task_groups.items():
            for group_id in numa_task_groups:
                if group_id == group.group_id:
                    task_group_numas.append(numa_id)
                    break
        group.numa = sorted(task_group_numas)

        # update socket
        task_group_sockets: list[int] = []
        for socked_id, socket_task_groups in self._socket_to_task_groups.items():
            for group_id in socket_task_groups:
                if group_id == group.group_id:
                    task_group_sockets.append(socked_id)
                    break
        group.socket = sorted(task_group_sockets)

        return True

    def _schedule_tasks_affinity_of_group(self, group: TaskGroup) -> bool:
        schedule_tasks = self._get_tasks_to_schedule_for_group(group)
        schedule_task_num = len(schedule_tasks)
        group_cluster_num = len(group.cluster)

        if schedule_task_num == 0:
            print(f"no task need to schedule in group [{group.group_id}]")
            return True

        group_cpus = group.cpus.to_list()
        if len(group_cpus) == 0:
            print(f"group[{group.group_id}] has no available CPU, cannot schedule.")
            return False

        for task in schedule_tasks:
            self._update_task_affinity(task, group.socket, group.numa, group.cluster, group.cpus.to_list())

        return True

    def _get_tasks_to_schedule_for_group(self, group: TaskGroup) -> list[Task]:
        schedule_tasks: list[Task] = []
        for task in group.get_all_tasks():
            if task.cpus.count() == 0:  # cpu尚未分配
                schedule_tasks.append(task)
        return schedule_tasks

    def _update_task_group_with_high_prio_tasks(self, group: TaskGroup) -> None:
        for task in group.get_high_prio_tasks():
            if task.cpus.count() != 0 and task.do_isolate:
                group.isolate_numa.extend(task.numa)
                group.isolate_cluster.extend(task.cluster)
                group.isolate_cpus.set_list(task.cpus.to_list())
                group.isolate_numa = sorted(set(group.isolate_numa))
                group.isolate_cluster = sorted(set(group.isolate_cluster))

    def _update_task_affinity(
        self,
        task: Task,
        socket: list[int],
        numa: list[int],
        cluster: list[int],
        cpus: list[int],
    ):
        task.socket = list(socket)
        task.numa = list(numa)
        task.cluster = list(cluster)
        task.assign_cpu(cpus)

    def _get_cpu_list_of_clusters(self, clusters: list[int]) -> list:
        cpu_list: list = []
        for cluster_id in clusters:
            cluster_domain = self.domain.get_cluster_domain(cluster_id)
            if cluster_domain is None:
                print(f"cluster domain [{cluster_id}] not found")
                continue
            cpu_list.extend(cluster_domain.cpus.to_list())
        return sorted(cpu_list)

    def get_background_task_cpus(self) -> list:
        """
        获取背景任务应绑定的CPU列表。
        优先使用空闲NUMA的CPU，若无空闲NUMA则使用各NUMA预留给系统任务的cluster。
        """

        used_numas = set()
        for _, group in self.task.groups.items():
            for task in group.get_all_tasks():
                if task.numa:
                    used_numas.update(task.numa)
        all_numas = set(self.domain.get_all_numas_id())
        unused_numas = all_numas - used_numas

        if unused_numas:
            cpus = []
            for numa_id in sorted(unused_numas):
                numa = self.domain.get_numa_domain(numa_id)
                if numa:
                    cpus.extend(numa.cpus.to_list())
            result = sorted(set(cpus))
            if result:
                print(
                    f"Using unused NUMA CPUs: {utils.compress_continuous(result)} "
                    f"numa={sorted(unused_numas)}"
                )
                return result

        background_cpus = []
        for numa_id in sorted(self._numa_to_background_clusters):
            clusters = self._numa_to_background_clusters[numa_id]
            background_cpus.extend(self._get_cpu_list_of_clusters(clusters))

        result = sorted(set(background_cpus))
        if result:
            print(
                f"Using reserved clusters from each NUMA: {utils.compress_continuous(result)} "
                f"(numa_reserved={dict(sorted(self._numa_to_background_clusters.items()))})"
            )
        return result

    def _alloc_cpus_for_background_tasks(self) -> None:
        normal_cpus = self.get_background_task_cpus()
        if normal_cpus:
            self.task.background_tasks_cpus = normal_cpus
            print(f"Allocated CPUs: {utils.compress_continuous(normal_cpus)}")
        else:
            print("No target CPUs found for background tasks")
