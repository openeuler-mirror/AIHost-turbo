from __future__ import annotations
from collections import defaultdict

from a_sched.affinity_domain import AffinityDomainManager
from a_sched.task import TaskManager
from a_sched.task.npu_task import NpuTask
from a_sched.task.task_base import PriorityLevel
from a_sched.config import AffinityConfig
import a_sched.config as cfg
from a_sched.strategy.scheduler_base import SchedulerBase
from a_sched.strategy.scheduler_factory import SchedulerFactory
from a_sched import utils


@SchedulerFactory.register("hbs_npu_affinity_isolate_cluster")
class HbsNpuAffinityIsolateCluster(SchedulerBase):
    """
    基于NPU亲和的分层均衡亲和调度策略

    适用场景：A5场景，设置了npu_affinity时的调度策略。优先将关键线程、NPU相关的进程绑定在NPU所亲和的NUMA上
    """

    def __init__(self, config: AffinityConfig, domain: AffinityDomainManager, task: TaskManager):
        super().__init__(config=config, domain=domain, task=task)

        # NPU亲和拓扑信息
        self._npu_to_numa: dict[int, int] = {}  # npu_id -> numa_id
        self._numa_to_npus: dict[int, list[int]] = defaultdict(list)  # numa_id -> [npu_ids]
        self._npu_to_socket: dict[int, int] = {}  # npu_id -> socket_id
        self._socket_to_npus: dict[int, list[int]] = defaultdict(list)  # socket_id -> [npu_ids]
        self._numa_to_socket: dict[int, int] = {}  # numa_id -> socket_id（拓扑事实，与NPU无关）

        # 调度分配信息
        self._socket_to_task_groups: dict[int, list[int]] = defaultdict(list)  # socket_id -> [task_group_ids]
        self._task_group_to_numas: dict[int, list[int]] = {}  # task_group_id -> 组内所有NPU亲和NUMA集合
        self._task_group_to_sockets: dict[int, list[int]] = {}  # task_group_id -> 组内所有NPU亲和socket集合

        # 背景任务预留cluster
        self._numa_to_background_clusters: dict[int, list[int]] = defaultdict(list)

        # 已使用的cluster跟踪（独占或被进程共享使用后均标记，组间互斥）
        self._used_clusters: set[int] = set()

    def schedule(self) -> None:
        """
        统一调度入口
        """
        # 1. 初始化NPU亲和拓扑信息
        self._init_npu_affinity_topo()

        # 2. 校验亲和组约束
        self._validate_affinity_group_constraints()

        # 3. 为背景任务分配cluster
        self._assign_clusters_for_background_tasks()

        # 4. 将亲和组分配到对应的socket（基于NPU亲和关系）
        self._distribute_task_groups_to_sockets()

        # 5. 为每个亲和组分配cluster
        for group_id in self._get_task_groups_to_schedule():
            self._assign_clusters_for_task_group(group_id)

    def _init_npu_affinity_topo(self) -> None:
        """
        初始化NPU亲和拓扑信息
        获取每个NPU亲和的NUMA和Socket
        """
        # 获取NPU到NUMA的亲和关系
        self._npu_to_numa = utils.get_npu_topo_numa_affinity()

        if not self._npu_to_numa:
            raise RuntimeError("Failed to get NPU numa affinity information")

        # 建立反向映射和Socket映射
        for npu_id, numa_id in self._npu_to_numa.items():
            # NUMA到NPU的反向映射
            self._numa_to_npus[numa_id].append(npu_id)

            # 获取该NUMA所属的Socket
            numa_domain = self.domain.get_numa_domain(numa_id)
            if numa_domain is None:
                raise RuntimeError(f"NUMA domain {numa_id} not found")

            # 找到该NUMA的父Socket
            if numa_domain.parent is None:
                raise RuntimeError(f"Cannot find socket for NUMA {numa_id}")
            socket_id = numa_domain.parent.domain_id

            self._npu_to_socket[npu_id] = socket_id
            self._socket_to_npus[socket_id].append(npu_id)
            self._numa_to_socket[numa_id] = socket_id

        # 对每个列表进行排序
        for numa_id in self._numa_to_npus:
            self._numa_to_npus[numa_id] = sorted(self._numa_to_npus[numa_id])

        for socket_id in self._socket_to_npus:
            self._socket_to_npus[socket_id] = sorted(self._socket_to_npus[socket_id])

        print("NPU affinity topo initialized:")
        print(f"  NPU -> NUMA: {self._npu_to_numa}")
        print(f"  NPU -> Socket: {self._npu_to_socket}")

    def _validate_affinity_group_constraints(self) -> None:
        """
        校验亲和组约束：
        1. 亲和组必须绑定NPU
        2. 一个进程只能绑定一个NPU，一个NPU也只能被一个进程绑定
        """
        for group_id in self._get_task_groups_to_schedule():
            group = self.task.get_group(group_id)
            if group is None:
                continue

            # 检查亲和组是否绑定NPU
            if not group.npu_tasks:
                raise RuntimeError(f"Task group {group_id} must bind to NPU in npu_affinity mode")

            # 校验每个NPU都有亲和拓扑信息
            for npu_id in group.npu_tasks.keys():
                if npu_id not in self._npu_to_numa:
                    raise RuntimeError(f"NPU {npu_id} affinity information not found")

            # 一个进程只能绑定一个NPU，一个NPU也只能被一个进程绑定
            bind_pids: list[int] = []
            for npu_task in group.npu_tasks.values():
                if npu_task.bind_pid is not None:
                    bind_pids.append(npu_task.bind_pid)
            if len(bind_pids) != len(set(bind_pids)):
                raise RuntimeError(
                    f"Task group {group_id}: one process is bound to multiple NPUs, "
                    f"which violates the one-process-one-NPU constraint, bind_pids={bind_pids}"
                )

    def _assign_clusters_for_background_tasks(self) -> None:
        """
        为背景任务分配clusters

        策略：
        1. 优先使用空闲numa（所在socket没有分配NPU亲和组)；
        2. 若没有空闲numa，则在非NPU亲和的numa中预留20%的cluster。
        """
        npu_affinity_numas = set(self._npu_to_numa.values())
        npu_sockets = {self._npu_to_socket[npu_id] for npu_id in self._npu_to_numa}

        print("Assigning clusters for background tasks:")
        print(f"  NPU affinity NUMAs: {sorted(npu_affinity_numas)}")
        print(f"  NPU affinity sockets: {sorted(npu_sockets)}")

        # 划分：空闲numa（socket无NPU亲和） vs 需预留numa（有NPU亲和socket中的非NPU亲和numa）
        unused_numas: list[int] = []
        to_reserve_numas: list[int] = []
        for socket in self.domain.socket_domains:
            socket_id = socket.domain_id
            socket_numas = socket.get_all_children_id()
            if socket_id not in npu_sockets:
                # 该socket无NPU亲和组，其numa全部空闲，直接用全部cpu
                unused_numas.extend(socket_numas)
                print(f"  Socket[{socket_id}]: no NPU affinity, NUMAs {sorted(socket_numas)} are idle")
            else:
                # 该socket有NPU亲和组，非NPU亲和的numa需预留20%cluster
                non_npu_numas_in_socket = sorted([n for n in socket_numas if n not in npu_affinity_numas])
                to_reserve_numas.extend(non_npu_numas_in_socket)
                print(f"  Socket[{socket_id}]: reserve from non-NPU affinity NUMAs {non_npu_numas_in_socket}")

        numa_to_background_clusters = self._do_assign_cpus_for_background_tasks(
            unused_numas=sorted(unused_numas),
            to_resrve_numas=sorted(to_reserve_numas),
        )

        # 将预留结果回填到hbs状态
        for numa_id, clusters in numa_to_background_clusters.items():
            self._numa_to_background_clusters[numa_id] = clusters
            self._used_clusters.update(clusters)

    def _distribute_task_groups_to_sockets(self) -> None:
        """
        将亲和组分配到对应的socket
        优先按亲和组绑定的NPU所亲和的socket分配
        """
        task_groups = self._get_task_groups_to_schedule()

        if not task_groups:
            raise RuntimeError("No valid task group found to distribute")

        for group_id in task_groups:
            group = self.task.get_group(group_id)
            if group is None or not group.npu_tasks:
                continue

            group_numas = sorted({self._npu_to_numa[n] for n in group.npu_tasks.keys()})
            group_sockets = sorted({self._npu_to_socket[n] for n in group.npu_tasks.keys()})

            # 将组分配到其涉及的所有socket（每个socket都记录该组）
            for socket_id in group_sockets:
                self._socket_to_task_groups[socket_id].append(group_id)
            self._task_group_to_numas[group_id] = group_numas
            self._task_group_to_sockets[group_id] = group_sockets

        print("Task groups distributed to sockets:")
        for socket_id, groups in self._socket_to_task_groups.items():
            print(f"  Socket[{socket_id}]: groups {groups}")

    def _assign_clusters_for_task_group(self, group_id: int) -> None:
        """
        按config.npu_process_cluster_mode布局策略为亲和组分配clusters

        策略：
        1. 为NPU的高优先级线程（acl_thread、release_thread、rt_recycle及对应高优先级线程）分配独占cluster
        2. 亲和组中npu进程和未绑定的进程和线程范围绑定在npu亲核的numa中的其它cluster上
        """
        group = self.task.get_group(group_id)
        if group is None:
            return

        group_numas = self._task_group_to_numas.get(group_id)
        group_sockets = self._task_group_to_sockets.get(group_id)
        if not group_numas or not group_sockets:
            print(f"[Warning] Cannot find NUMA/socket for task group {group_id}")
            return

        print(f"\nAssigning clusters for task group {group_id}:")
        print(
            f"  NPU affinity NUMAs: {group_numas}, Sockets: {group_sockets}, "
            f"npu_process_cluster_mode: {self.config.npu_process_cluster_mode}"
        )

        # 1. 按npu_process_cluster_mode布局策略，为每个NPU的高优先级线程（acl_thread、release_thread、
        #    rt_recycle及对应高优先级线程）分配独占cluster
        npu_to_cluster: dict[int, int] = {}  # npu_id -> 选中的cluster（colocated/shared布局下进程复用）
        if self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_SHARED:
            # shared布局下，批量处理所有NPU的cluster分配（支持多NPU共享cluster，按各NPU自身亲和NUMA分组）
            npu_to_cluster = self._npu_process_cluster_mode_shared_high_prio(group_id=group_id)
        else:
            for npu_id, npu_task in group.npu_tasks.items():
                if self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_ISOLATED:
                    self._npu_process_cluster_mode_isolated_high_prio(
                        group_id=group_id,
                        npu_id=npu_id,
                        npu_task=npu_task,
                    )
                elif self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_COLOCATED:
                    cluster = self._npu_process_cluster_mode_colocated_high_prio(
                        group_id=group_id,
                        npu_id=npu_id,
                        npu_task=npu_task,
                    )
                    npu_to_cluster[npu_id] = cluster
                else:
                    raise RuntimeError(
                        f"Task group {group_id}: unsupported npu_process_cluster_mode strategy "
                        f"'{self.config.npu_process_cluster_mode}'"
                    )

        # 2. 按npu_process_cluster_mode布局策略，为绑定NPU的进程及未绑定NPU的进程/线程分配cluster
        npu_bind_pids = [npu_task.bind_pid for npu_task in group.npu_tasks.values() if npu_task.bind_pid is not None]
        npu_bind_processes = [task for task in group.process_tasks.values() if task.task_id in npu_bind_pids]
        # 亲和组中未绑定NPU的进程和非关键线程
        other_processes = [task for task in group.process_tasks.values() if task.task_id not in npu_bind_pids]
        other_threads = [
            thread
            for thread in group.thread_tasks.values()
            if thread.bind_npu is None and thread.priority != PriorityLevel.HIGH
        ]

        shared_tasks = npu_bind_processes + other_processes + other_threads
        shared_clusters: list[int] = []
        if shared_tasks:
            if self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_ISOLATED:
                shared_clusters = self._npu_process_cluster_mode_isolated(
                    group_id=group_id,
                    tasks=shared_tasks,
                )
            elif self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_COLOCATED:
                shared_clusters = self._npu_process_cluster_mode_colocated(
                    group_id=group_id,
                    npu_bind_processes=npu_bind_processes,
                    other_processes=other_processes,
                    other_threads=other_threads,
                    npu_to_cluster=npu_to_cluster,
                )
            elif self.config.npu_process_cluster_mode == cfg.NPU_PROCESS_CLUSTER_MODE_SHARED:
                # shared布局：绑定进程绑NPU亲和numa其它未使用cluster，未绑进程范围绑到绑定进程cluster并集(共享池)
                shared_clusters = self._npu_process_cluster_mode_shared(
                    group_id=group_id,
                    tasks=shared_tasks,
                )
            else:
                raise RuntimeError(
                    f"Task group {group_id}: unsupported npu_process_cluster_mode strategy "
                    f"'{self.config.npu_process_cluster_mode}'"
                )

        # 更新task group的亲和信息
        # 注意：shared_clusters 可能降级到同 socket 的兄弟 NUMA，需按 cluster 实际所属
        # NUMA/socket 推导 group.numa/group.socket，避免与 group.cluster / group.cpus 归属不一致。
        if shared_tasks and shared_clusters:
            group.numa = sorted(self.domain.get_numas_of_clusters(shared_clusters))
            group.socket = sorted(self.domain.get_sockets_of_clusters(shared_clusters))
        else:
            group.numa = sorted(group_numas)
            group.socket = sorted(group_sockets)
        if shared_tasks and shared_clusters:
            group.cluster = sorted(shared_clusters)
            group.cpus.set_list(self.domain.get_cpus_of_clusters(clusters=shared_clusters))

    def _npu_process_cluster_mode_isolated(
        self,
        group_id: int,
        tasks: list,
    ) -> list[int]:
        """isolated布局：绑定NPU进程按其绑定NPU各自亲和NUMA取其它未使用cluster（同NUMA的进程
        共享该NUMA可用cluster，降级同socket兄弟NUMA）；未绑定NPU的进程/非关键线程范围绑到
        组内所有绑定进程所在cluster的cpu核并集。
        """
        group = self.task.get_group(group_id)
        group_numas = self._task_group_to_numas.get(group_id, [])

        # bind_pid -> npu_id 反查，用于按各自NPU亲和NUMA分组绑定进程
        pid_to_npu: dict[int, int] = {}
        for npu_id, npu_task in group.npu_tasks.items() if group is not None else []:
            if npu_task.bind_pid is not None:
                pid_to_npu[npu_task.bind_pid] = npu_id

        bind_tasks = [t for t in tasks if t.task_id in pid_to_npu]
        unbound_tasks = [t for t in tasks if t.task_id not in pid_to_npu]

        all_clusters: set[int] = set()
        # 1. 绑定进程：按各自NPU亲和NUMA/socket分组取其它未使用cluster
        if bind_tasks:
            pid_groups: dict[tuple[int, int], list] = {}
            for t in bind_tasks:
                npu_id = pid_to_npu[t.task_id]
                nid = self._npu_to_numa[npu_id]
                sck = self._npu_to_socket[npu_id]
                pid_groups.setdefault((nid, sck), []).append(t)
            for (nid, sck), procs in pid_groups.items():
                bind_clusters = self._acquire_clusters_for_processes(numa_id=nid, socket_id=sck)
                if not bind_clusters:
                    raise RuntimeError(
                        f"Task group {group_id}: no available cluster for bound processes/threads "
                        f"(numa={nid}, socket={sck})"
                    )
                self._assign_clusters_cpus_to_tasks(clusters=bind_clusters, tasks=procs)
                self._used_clusters.update(bind_clusters)
                all_clusters.update(bind_clusters)
        # 2. 未绑定进程/非关键线程：范围绑到所有绑定进程所在cluster的核并集
        if unbound_tasks:
            if all_clusters:
                unbound_clusters = sorted(all_clusters)
            else:
                # 无绑定进程：用组内所有NPU亲和NUMA的可用cluster并集
                # 按NUMA分组（同NUMA的NPU共享其亲和NUMA的可用cluster）
                unbound_clusters: list[int] = []
                for nid in group_numas:
                    sck = self._numa_to_socket.get(nid)
                    if sck is None:
                        continue
                    unbound_clusters.extend(self._acquire_clusters_for_processes(numa_id=nid, socket_id=sck))
                unbound_clusters = sorted(set(unbound_clusters))
            bind_cpus_pool = self.domain.get_cpus_of_clusters(clusters=unbound_clusters)
            if not bind_cpus_pool:
                raise RuntimeError(
                    f"Task group {group_id}: no available cluster for unbound processes/threads (numas={group_numas})"
                )
            self._assign_cpus_to_tasks(cpus=bind_cpus_pool, tasks=unbound_tasks)
            self._used_clusters.update(unbound_clusters)
            all_clusters.update(unbound_clusters)

        return sorted(all_clusters)

    def _npu_process_cluster_mode_colocated_high_prio(
        self,
        group_id: int,
        npu_id: int,
        npu_task,
    ) -> int:
        """
        colocated布局策略下，为NPU的高优先级线程（acl_thread、release_thread、rt_recycle
        及对应高优先级线程）分配一个独占cluster。npu进程复用该cluster中高优先级线程未使用的核

        按该NPU自身亲和NUMA/socket选取，降级同socket其它numa，仍无则失败。

        Returns:
            选中的cluster id
        """
        numa_id = self._npu_to_numa[npu_id]
        socket_id = self._npu_to_socket[npu_id]
        cluster = self._acquire_one_cluster(numa_id=numa_id, socket_id=socket_id)
        if cluster is None:
            raise RuntimeError(
                f"Task group {group_id}, NPU[{npu_id}]: no available cluster for high priority threads "
                f"(acl/release/rt_recycle), numa={numa_id}, socket={socket_id}"
            )

        # 获取该cluster的CPU
        cluster_cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
        if not cluster_cpus:
            raise RuntimeError(f"Task group {group_id}, NPU[{npu_id}]: cluster {cluster} has no CPUs")

        # 分配CPU给NPU task（内部会为acl_thread/release_thread/rt_recycle及高优先级线程分核）
        # 按cluster_cpus实际所属NUMA/socket推导，避免降级到兄弟NUMA时与cluster归属不一致
        self._update_task_affinity(npu_task, cluster_cpus)

        # colocated布局下，进程会复用该cluster的剩余核，cluster整体标记为已使用
        self._used_clusters.add(cluster)

        print(
            f"  NPU[{npu_id}]: assigned cluster {cluster} (numa={npu_task.numa}) "
            f"for high priority threads, cpus={utils.compress_continuous(cluster_cpus)}"
        )
        return cluster

    @staticmethod
    def _solve_min_k(npu_num: int, cluster_num: int, min_capacity: int) -> int:
        """求最小的每cluster容纳数k，使 ceil(npu_num/k) <= cluster_num 且 k <= min_capacity。

        优先均衡：所有被分配cluster容纳相同数目k（最后一个可能少），k尽可能小。
        均衡分配无法满足时退化为按容量尽量塞（k=min_capacity，不严格均衡）。
        """
        k = 1
        while k <= min_capacity:
            need_clusters = (npu_num + k - 1) // k  # ceil(npu_num / k)
            if need_clusters <= cluster_num:
                break
            k += 1
        return min(k, min_capacity)

    def _npu_process_cluster_mode_shared_high_prio(
        self,
        group_id: int,
    ) -> dict[int, int]:
        """
        shared布局策略下，为亲和组内所有NPU的高优先级线程（acl_thread、release_thread、
        rt_recycle及对应高优先级线程）分配cluster。

        多NUMA/多socket场景按各NPU自身亲和NUMA分组，每组各自独立决策（支持亲和到不同NUMA）：
        1. 若该numa的可用cluster数目(降级同socket其它numa) >= 本组NPU数目，每个NPU独占一个cluster
        2. 若cluster数目 < 本组NPU数目，但单个cluster可满足多个NPU关键线程所需核数，
           多个NPU共享cluster；优先保障被分配cluster容纳的NPU数目均衡，其次每个cluster
           容纳的NPU数目尽可能少
        3. 若仍不满足则分配失败

        Returns:
            dict: npu_id -> 选中的cluster id
        """
        group = self.task.get_group(group_id)
        npu_to_cluster: dict[int, int] = {}
        # 按各NPU自身亲和NUMA/socket分组，每组独立决策
        numas: dict[tuple[int, int], list[int]] = {}
        for npu_id in sorted(group.npu_tasks.keys()):
            nid = self._npu_to_numa[npu_id]
            sck = self._npu_to_socket[npu_id]
            numas.setdefault((nid, sck), []).append(npu_id)
        for (nid, sck), npu_ids in numas.items():
            self._assign_shared_high_prio_for_numa(
                group_id=group_id,
                numa_id=nid,
                socket_id=sck,
                npu_ids=npu_ids,
                npu_to_cluster=npu_to_cluster,
            )
        return npu_to_cluster

    def _assign_shared_high_prio_for_numa(
        self,
        group_id: int,
        numa_id: int,
        socket_id: int,
        npu_ids: list[int],
        npu_to_cluster: dict[int, int],
    ) -> None:
        """
        shared布局策略下，为亲和到同一NUMA的一组NPU的高优先级线程分配cluster（独立决策）。

        分配策略：
        1. 若可用cluster数目 >= 本组NPU数目，每个NPU独占一个cluster
        2. 若cluster数目 < 本组NPU数目，但单个cluster可满足多个NPU关键线程所需核数，
           多个NPU共享cluster；优先均衡，其次每个cluster容纳的NPU数目尽可能少
        3. 若仍不满足则分配失败
        """
        group = self.task.get_group(group_id)
        npu_num = len(npu_ids)

        # 收集该NUMA中可用cluster（降级到同socket其它numa）
        available_clusters = self._acquire_clusters_for_processes(numa_id=numa_id, socket_id=socket_id)
        if not available_clusters:
            raise RuntimeError(
                f"Task group {group_id}: no available cluster for high priority threads "
                f"(acl/release/rt_recycle), numa={numa_id}, socket={socket_id}"
            )

        cluster_num = len(available_clusters)
        npu_need_cpus = {npu_id: group.npu_tasks[npu_id].min_cpu for npu_id in npu_ids}
        cluster_cpu_size = {c: len(self.domain.get_cpus_of_clusters(clusters=[c])) for c in available_clusters}

        # 策略1：cluster数目 >= NPU数目，每个NPU独占一个cluster
        if cluster_num >= npu_num:
            for i, npu_id in enumerate(npu_ids):
                cluster = available_clusters[i]
                self._assign_shared_high_prio_cluster(
                    group_id=group_id,
                    npu_id=npu_id,
                    npu_task=group.npu_tasks[npu_id],
                    cluster=cluster,
                )
                npu_to_cluster[npu_id] = cluster
            return

        # 策略2：cluster数目 < NPU数目，多NPU共享cluster
        # 优先保障被分配cluster容纳的NPU数目均衡，其次每个cluster容纳的NPU数目尽可能少
        # 即：找最小的每cluster容纳数k，使得 ceil(NPU数/k) <= cluster数 且 k <= 每cluster容量，
        # 使用 ceil(NPU数/k) 个cluster，每个容纳k个（最后一个容纳余数）
        max_cpus_per_npu = max(npu_need_cpus.values()) if npu_need_cpus else 1
        # 每个cluster能容纳的NPU数 = cluster核数 // 单NPU所需核数（用最大所需核数保守估计）
        cluster_capacity = {c: max(1, cluster_cpu_size[c] // max_cpus_per_npu) for c in available_clusters}
        total_capacity = sum(cluster_capacity.values())

        if total_capacity < npu_num:
            # 策略3：仍不满足，分配失败
            raise RuntimeError(
                f"Task group {group_id}: cannot satisfy {npu_num} NPUs with {cluster_num} clusters "
                f"(total capacity {total_capacity} < npu num {npu_num}), numa={numa_id}, socket={socket_id}"
            )

        # 寻找最小的每cluster容纳数k，使 ceil(npu_num/k) <= cluster_num 且 k <= min(cluster_capacity)
        # 优先均衡：所有被分配cluster容纳相同数目k（最后一个可能少），k尽可能小
        min_capacity = min(cluster_capacity.values())
        k = self._solve_min_k(npu_num, cluster_num, min_capacity)

        # 使用前 need_clusters 个cluster，每个容纳k个NPU（最后一个容纳余数）
        need_clusters = (npu_num + k - 1) // k
        used_clusters = available_clusters[:need_clusters]
        remaining_npus = list(npu_ids)

        for cluster in used_clusters:
            if not remaining_npus:
                break
            # 该cluster容纳的NPU数（最后一个可能少于k）
            take = min(k, len(remaining_npus))
            take_npus = remaining_npus[:take]
            remaining_npus = remaining_npus[take:]

            # 该cluster的核分配给这些NPU（每个NPU按min_cpu取核）
            cluster_cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
            cpu_start = 0
            for npu_id in take_npus:
                need = npu_need_cpus[npu_id]
                npu_cpus = cluster_cpus[cpu_start : cpu_start + need]
                cpu_start += need
                self._assign_shared_high_prio_cluster(
                    group_id=group_id,
                    npu_id=npu_id,
                    npu_task=group.npu_tasks[npu_id],
                    cluster=cluster,
                    cpus=npu_cpus,
                )
                npu_to_cluster[npu_id] = cluster

        # 若仍有剩余NPU（均衡分配无法完全容纳），继续往有剩余容量的cluster塞
        for npu_id in remaining_npus:
            need = npu_need_cpus[npu_id]
            placed = False
            for cluster in available_clusters:
                used_in_cluster = sum(npu_need_cpus[n] for n, c in npu_to_cluster.items() if c == cluster)
                if used_in_cluster + need <= cluster_cpu_size[cluster]:
                    cluster_cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
                    npu_cpus = cluster_cpus[used_in_cluster : used_in_cluster + need]
                    self._assign_shared_high_prio_cluster(
                        group_id=group_id,
                        npu_id=npu_id,
                        npu_task=group.npu_tasks[npu_id],
                        cluster=cluster,
                        cpus=npu_cpus,
                    )
                    npu_to_cluster[npu_id] = cluster
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"Task group {group_id}, NPU[{npu_id}]: cannot find cluster with enough cpus "
                    f"({need} needed) for high priority threads, numa={numa_id}, socket={socket_id}"
                )

    def _assign_shared_high_prio_cluster(
        self,
        group_id: int,
        npu_id: int,
        npu_task,
        cluster: int,
        cpus: list[int] | None = None,
    ) -> None:
        """
        将指定cluster（或指定cpus）分配给NPU的高优先级线程，并设置亲和信息
        """
        if cpus is None:
            cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
        if not cpus:
            raise RuntimeError(f"Task group {group_id}, NPU[{npu_id}]: cluster {cluster} has no CPUs")

        # 按cpus实际所属NUMA/socket推导，避免降级到兄弟NUMA时与cluster归属不一致
        self._update_task_affinity(npu_task, cpus)

        # 标记该cluster为已使用（高优先级线程已占用部分核，进程不再复用该cluster）
        self._used_clusters.add(cluster)

        print(
            f"  NPU[{npu_id}]: assigned cluster {cluster} (numa={npu_task.numa}) "
            f"for high priority threads, cpus={utils.compress_continuous(cpus)}"
        )

    def _npu_process_cluster_mode_shared(
        self,
        group_id: int,
        tasks: list,
    ) -> list[int]:
        """shared布局：绑定NPU进程按其绑定NPU各自亲和NUMA取其它未使用cluster（同NUMA的进程
        共享该NUMA可用cluster，降级同socket兄弟NUMA）；未绑定NPU的进程/非关键线程范围绑到
        组内所有绑定进程所在cluster的cpu核并集（共享池）。无绑定进程时
        退化为组所有NPU亲和NUMA可用cluster并集。
        """
        group = self.task.get_group(group_id)
        group_numas = self._task_group_to_numas.get(group_id, [])

        pid_to_npu: dict[int, int] = {}
        for npu_id, npu_task in group.npu_tasks.items() if group is not None else []:
            if npu_task.bind_pid is not None:
                pid_to_npu[npu_task.bind_pid] = npu_id

        bind_tasks = [t for t in tasks if t.task_id in pid_to_npu]
        unbound_tasks = [t for t in tasks if t.task_id not in pid_to_npu]

        all_clusters: set[int] = set()
        # 1. 绑定进程：按各自NPU亲和NUMA/socket分组取其它未使用cluster
        if bind_tasks:
            pid_groups: dict[tuple[int, int], list] = {}
            for t in bind_tasks:
                npu_id = pid_to_npu[t.task_id]
                nid = self._npu_to_numa[npu_id]
                sck = self._npu_to_socket[npu_id]
                pid_groups.setdefault((nid, sck), []).append(t)
            for (nid, sck), procs in pid_groups.items():
                bind_clusters = self._acquire_clusters_for_processes(numa_id=nid, socket_id=sck)
                if not bind_clusters:
                    raise RuntimeError(
                        f"Task group {group_id}: no available cluster for bound processes/threads "
                        f"(numa={nid}, socket={sck})"
                    )
                self._assign_clusters_cpus_to_tasks(clusters=bind_clusters, tasks=procs)
                self._used_clusters.update(bind_clusters)
                all_clusters.update(bind_clusters)
        # 2. 未绑定进程/非关键线程：范围绑到所有绑定进程所在cluster的核并集（共享池）
        if unbound_tasks:
            if all_clusters:
                unbound_clusters = sorted(all_clusters)
            else:
                # 无绑定进程：用组内所有NPU亲和NUMA的可用cluster并集
                unbound_clusters: list[int] = []
                for nid in group_numas:
                    sck = self._numa_to_socket.get(nid)
                    if sck is None:
                        continue
                    unbound_clusters.extend(self._acquire_clusters_for_processes(numa_id=nid, socket_id=sck))
                unbound_clusters = sorted(set(unbound_clusters))
            bind_cpus_pool = self.domain.get_cpus_of_clusters(clusters=unbound_clusters)
            if not bind_cpus_pool:
                raise RuntimeError(
                    f"Task group {group_id}: no available cluster for unbound processes/threads (numas={group_numas})"
                )
            self._assign_cpus_to_tasks(cpus=bind_cpus_pool, tasks=unbound_tasks)
            self._used_clusters.update(unbound_clusters)
            all_clusters.update(unbound_clusters)

        return sorted(all_clusters)

    def _npu_process_cluster_mode_colocated(
        self,
        group_id: int,
        npu_bind_processes: list,
        other_processes: list,
        other_threads: list,
        npu_to_cluster: dict[int, int],
    ) -> list[int]:
        """
        colocated布局策略：绑定该NPU的进程范围绑定到上述高优先级线程所在cluster中的
        其它未使用的核；亲和组中其它未绑定NPU的进程/线程共享所有NPU高优先级线程所在cluster的
        剩余核并集（共享池）。

        Args:
            npu_bind_processes: 绑定NPU的进程列表，与npu_to_cluster的key顺序对应
            other_processes: 未绑定NPU的进程列表
            other_threads: 未绑定NPU的普通优先级线程列表
            npu_to_cluster: npu_id -> 高优先级线程选中的cluster id
        """
        group = self.task.get_group(group_id)
        all_clusters: set[int] = set()

        # 绑定NPU的进程按其绑定的npu索引
        pid_to_npu: dict[int, int] = {}
        for npu_id, npu_task in group.npu_tasks.items() if group is not None else []:
            if npu_task.bind_pid is not None:
                pid_to_npu[npu_task.bind_pid] = npu_id

        # 1. 绑定NPU的进程：各自绑定到所绑NPU高优先级线程所在cluster的剩余核
        for proc in npu_bind_processes:
            npu_id = pid_to_npu.get(proc.task_id)
            if npu_id is None:
                continue
            cluster = npu_to_cluster.get(npu_id)
            if cluster is None:
                continue
            npu_task = group.npu_tasks.get(npu_id)
            remain_cpus = self._get_remain_cpus_in_cluster(cluster=cluster, npu_task=npu_task)
            if not remain_cpus:
                raise RuntimeError(
                    f"Task group {group_id}, NPU[{npu_id}]: no remain cpus in cluster {cluster} for bound processes"
                )
            self._assign_cpus_to_tasks(cpus=remain_cpus, tasks=[proc])
            all_clusters.add(cluster)

        # 2. 未绑定NPU的进程/线程：共享所有NPU高优先级线程所在cluster的剩余核并集（共享池）
        other_tasks = other_processes + other_threads
        if other_tasks:
            shared_cpus: list[int] = []
            for npu_id, cluster in npu_to_cluster.items():
                npu_task = group.npu_tasks.get(npu_id)
                shared_cpus.extend(self._get_remain_cpus_in_cluster(cluster=cluster, npu_task=npu_task))
            shared_cpus = sorted(set(shared_cpus))
            if not shared_cpus:
                raise RuntimeError(
                    f"Task group {group_id}: no remain cpus in colocated clusters {sorted(npu_to_cluster.values())} "
                    f"for unbound processes/threads"
                )
            self._assign_cpus_to_tasks(cpus=shared_cpus, tasks=other_tasks)
            all_clusters.update(npu_to_cluster.values())

        return sorted(all_clusters)

    def _get_remain_cpus_in_cluster(self, cluster: int, npu_task) -> list[int]:
        """
        获取cluster中高优先级线程分配后剩余的核
        assign_cpu按顺序为acl/release/rt_recycle及高优先级线程各分配1个核，剩余即为cluster_cpus[min_cpu:]
        """
        cluster_cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
        if not cluster_cpus:
            return []
        used_num = npu_task.min_cpu if npu_task is not None else 0
        return cluster_cpus[used_num:]

    def _npu_process_cluster_mode_isolated_high_prio(
        self,
        group_id: int,
        npu_id: int,
        npu_task: NpuTask,
    ) -> None:
        """
        isolated布局策略下，为NPU的高优先级线程（acl_thread、release_thread、rt_recycle
        及对应高优先级线程）分配一个独占cluster。按该NPU自身亲和NUMA/socket选取，
        降级同socket其它numa，仍无则失败。
        """
        numa_id = self._npu_to_numa[npu_id]
        socket_id = self._npu_to_socket[npu_id]
        cluster = self._acquire_one_cluster(numa_id=numa_id, socket_id=socket_id)
        if cluster is None:
            raise RuntimeError(
                f"Task group {group_id}, NPU[{npu_id}]: no available cluster for high priority threads "
                f"(acl/release/rt_recycle), numa={numa_id}, socket={socket_id}"
            )

        # 获取该cluster的CPU
        cluster_cpus = self.domain.get_cpus_of_clusters(clusters=[cluster])
        if not cluster_cpus:
            raise RuntimeError(f"Task group {group_id}, NPU[{npu_id}]: cluster {cluster} has no CPUs")

        # 分配CPU给NPU task（内部会为acl_thread/release_thread/rt_recycle及高优先级线程分核）
        # 按cluster实际所属NUMA/socket推导，避免降级到兄弟NUMA时与cluster归属不一致
        actual_numas = self.domain.get_numas_of_clusters([cluster])
        actual_sockets = self.domain.get_sockets_of_clusters([cluster])
        npu_task.socket = actual_sockets
        npu_task.numa = actual_numas
        npu_task.cluster = [cluster]
        npu_task.assign_cpu(cluster_cpus)

        # 标记该cluster为已使用（独占）
        self._used_clusters.add(cluster)

        print(
            f"  NPU[{npu_id}]: assigned cluster {cluster} (numa={actual_numas}) "
            f"for high priority threads, cpus={utils.compress_continuous(cluster_cpus)}"
        )

    def _acquire_one_cluster(self, numa_id: int, socket_id: int) -> int | None:
        """
        按优先级获取一个可分配的cluster：
        1. NPU亲和numa中可分配的cluster
        2. 同socket其它numa中可分配的cluster
        3. 都没有则返回None
        """
        # 1. 优先从NPU亲和numa中选取
        clusters = self._get_available_clusters_in_numa(numa_id)
        if clusters:
            return clusters[0]

        # 2. NPU亲和numa中无可分配cluster，选择同socket其它numa的cluster
        same_socket_numas = self._get_sibling_numas_in_socket(socket_id, exclude=numa_id)
        for sib_numa in same_socket_numas:
            clusters = self._get_available_clusters_in_numa(sib_numa)
            if clusters:
                print(
                    f"    NPU affinity NUMA[{numa_id}] has no available cluster, "
                    f"fallback to sibling NUMA[{sib_numa}] in socket[{socket_id}]"
                )
                return clusters[0]

        # 3. 仍无可分配cluster
        return None

    def _acquire_clusters_for_processes(
        self,
        numa_id: int,
        socket_id: int,
    ) -> list[int]:
        """
        为进程/线程获取可分配的cluster范围：
        1. 优先使用NPU亲和numa中所有可分配的cluster
        2. 若NPU亲和numa中无可分配cluster，使用同socket其它numa中可分配的cluster
        3. 若仍无可分配cluster，返回空列表
        """
        # 1. 优先从NPU亲和numa中收集
        clusters = self._get_available_clusters_in_numa(numa_id)
        if clusters:
            return clusters

        # 2. NPU亲和numa中无可分配cluster，选择同socket其它numa的cluster
        same_socket_numas = self._get_sibling_numas_in_socket(socket_id, exclude=numa_id)
        for sib_numa in same_socket_numas:
            clusters = self._get_available_clusters_in_numa(sib_numa)
            if clusters:
                print(
                    f"    NPU affinity NUMA[{numa_id}] has no available cluster for processes, "
                    f"fallback to sibling NUMA[{sib_numa}] in socket[{socket_id}]"
                )
                return clusters

        # 3. 仍无可分配cluster
        return []

    def _get_available_clusters_in_numa(self, numa_id: int) -> list[int]:
        """
        获取指定numa中可分配的cluster
        可分配 = 未被使用（独占或共享）且未被背景预留
        """
        numa = self.domain.get_numa_domain(numa_id)
        if numa is None:
            return []

        all_clusters = set(numa.get_all_children_id())
        reserved = set(self._numa_to_background_clusters.get(numa_id, []))
        available = all_clusters - self._used_clusters - reserved
        return sorted(available)

    def _get_sibling_numas_in_socket(self, socket_id: int, exclude: int) -> list[int]:
        """
        获取同socket中除exclude外的其它numa
        """
        socket = self.domain.get_socket_domain(socket_id)
        if socket is None:
            return []
        numas = [nid for nid in socket.get_all_children_id() if nid != exclude]
        return sorted(numas)

