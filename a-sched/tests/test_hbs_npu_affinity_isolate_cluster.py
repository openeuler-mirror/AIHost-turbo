import sys
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

sys.modules["psutil"] = MagicMock()
sys.modules["psutil"].__version__ = "0.0.0"

from a_sched import config as cfg  # noqa: E402
from a_sched.config import AffinityConfig  # noqa: E402
from a_sched.strategy.hbs_npu_affinity_isolate_cluster import HbsNpuAffinityIsolateCluster  # noqa: E402
from a_sched.strategy import scheduler_factory as sf  # noqa: E402
from a_sched.task.process_task import ProcessTask  # noqa: E402
from a_sched.utils import AscendDeviceType  # noqa: E402


def _make_domain():
    """构造 mock domain：socket0->numa0(c0,c1),numa1(c2,c3); socket1->numa2(c4,c5),numa3(c6,c7)。"""
    domain = MagicMock()

    numa_clusters = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
    }

    def _get_numa_domain(numa_id):
        if numa_id not in numa_clusters:
            return None
        node_domain = MagicMock()
        node_domain.domain_id = numa_id
        node_domain.get_all_children_id.return_value = list(numa_clusters[numa_id])
        return node_domain

    def _get_socket_domain(socket_id):
        sd = MagicMock()
        sd.get_all_children_id.return_value = {0: [0, 1], 1: [2, 3]}.get(socket_id)
        return None if sd.get_all_children_id.return_value is None else sd

    domain.get_numa_domain.side_effect = _get_numa_domain
    domain.get_socket_domain.side_effect = _get_socket_domain

    domain.numa_domains = [_get_numa_domain(nid) for nid in numa_clusters]

    def _get_numas_of_clusters(clusters):
        numas = []
        for numa in domain.numa_domains:
            for cluster in clusters:
                if cluster in numa.get_all_children_id():
                    numas.append(numa.domain_id)
                    break
        return numas

    domain.get_numas_of_clusters.side_effect = _get_numas_of_clusters

    cluster_cpus = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
        5: [10, 11],
        6: [12, 13],
        7: [14, 15],
    }
    cpu_map = {}
    for numa_id, clusters in numa_clusters.items():
        socket_id = 0 if numa_id in (0, 1) else 1
        for cluster_id in clusters:
            for cpu in cluster_cpus[cluster_id]:
                cpu_map[cpu] = (socket_id, numa_id, cluster_id)

    def _get_cpus_of_clusters(clusters):
        cpus = []
        for cid in clusters or []:
            cpus.extend(cluster_cpus.get(cid, []))
        return sorted(cpus)

    def _get_numas_of_cpus(cpus):
        return sorted({cpu_map[c][1] for c in cpus if c in cpu_map})

    def _get_sockets_of_cpus(cpus):
        return sorted({cpu_map[c][0] for c in cpus if c in cpu_map})

    def _get_clusters_of_cpus(cpus):
        return sorted({cpu_map[c][2] for c in cpus if c in cpu_map})

    domain.get_cpus_of_clusters.side_effect = _get_cpus_of_clusters
    domain.get_numas_of_cpus.side_effect = _get_numas_of_cpus
    domain.get_sockets_of_cpus.side_effect = _get_sockets_of_cpus
    domain.get_clusters_of_cpus.side_effect = _get_clusters_of_cpus

    socket_to_numas = {0: [0, 1], 1: [2, 3]}

    def _make_socket_domain(socket_id):
        sd = MagicMock()
        sd.domain_id = socket_id
        sd.get_all_children_id.return_value = list(socket_to_numas[socket_id])
        return sd

    domain.socket_domains = [_make_socket_domain(sid) for sid in socket_to_numas]

    def _get_cpus_of_numas(numas):
        cpus = []
        for nid in numas or []:
            for cid in numa_clusters.get(nid, []):
                cpus.extend(cluster_cpus.get(cid, []))
        return sorted(cpus)

    domain.get_cpus_of_numas.side_effect = _get_cpus_of_numas

    return domain


def _make_scheduler():
    """绕过 __init__ 构造 HbsNpuAffinityIsolateCluster 实例并手工设置所需属性。"""
    sched = HbsNpuAffinityIsolateCluster.__new__(HbsNpuAffinityIsolateCluster)
    sched.config = AffinityConfig()
    sched.domain = _make_domain()
    sched.task = MagicMock()

    sched._numa_to_background_clusters = defaultdict(list)
    sched._used_clusters = set()
    sched._socket_to_task_groups = defaultdict(list)
    sched._task_group_to_numas = {}
    sched._task_group_to_sockets = {}
    sched._npu_to_numa = {}
    sched._npu_to_socket = {}
    sched._numa_to_socket = {0: 0, 1: 0, 2: 1, 3: 1}
    return sched


class TestSetScheduleStrategyConfig(unittest.TestCase):
    """UT for config.set_schedule_strategy"""

    def test_enable_npu_topo_affinity_sets_cluster_mode(self):
        c = AffinityConfig()
        c.set_schedule_strategy("cluster", npu_process_cluster_mode="isolated", enable_npu_topo_affinity=True)
        self.assertEqual(c.isolate, "cluster")
        self.assertEqual(c.npu_process_cluster_mode, "isolated")
        self.assertTrue(c.enable_npu_topo_affinity)

    def test_cluster_mode_set_unconditionally(self):
        c = AffinityConfig()
        c.set_schedule_strategy("cluster", npu_process_cluster_mode="isolated")
        self.assertEqual(c.isolate, "cluster")
        self.assertEqual(c.npu_process_cluster_mode, cfg.NPU_PROCESS_CLUSTER_MODE_ISOLATED)
        self.assertFalse(c.enable_npu_topo_affinity)

    def test_default_cluster_mode_is_isolated(self):
        c = AffinityConfig()
        self.assertEqual(c.npu_process_cluster_mode, cfg.NPU_PROCESS_CLUSTER_MODE_ISOLATED)
        self.assertFalse(c.enable_npu_topo_affinity)


class TestDecideOnNpuAffinity(unittest.TestCase):
    """UT for scheduler_factory.decide_on_npu_affinity"""

    def test_a5_returns_hbs_npu_affinity_isolate_cluster(self):
        with patch.object(sf.utils, "get_ascend_device_type", return_value=AscendDeviceType.A5):
            res = sf.decide_on_npu_affinity(config=AffinityConfig(), domain=MagicMock(), task=MagicMock())
        self.assertEqual(res, sf.HBS_NPU_AFFINITY_ISOLATE_CLUSTER)

    def test_non_a5_raises(self):
        with patch.object(sf.utils, "get_ascend_device_type", return_value=AscendDeviceType.A3):
            with self.assertRaises(RuntimeError):
                sf.decide_on_npu_affinity(config=AffinityConfig(), domain=MagicMock(), task=MagicMock())


class TestClusterFallbackHelpers(unittest.TestCase):
    """UT for cluster 降级分配辅助函数"""

    def test_get_available_clusters_in_numa(self):
        sched = _make_scheduler()
        self.assertEqual(sched._get_available_clusters_in_numa(0), [0, 1])

    def test_get_available_clusters_excludes_used(self):
        sched = _make_scheduler()
        sched._used_clusters.add(0)
        self.assertEqual(sched._get_available_clusters_in_numa(0), [1])

    def test_get_available_clusters_excludes_reserved(self):
        sched = _make_scheduler()
        sched._numa_to_background_clusters[0] = [1]
        self.assertEqual(sched._get_available_clusters_in_numa(0), [0])

    def test_get_sibling_numas_in_socket(self):
        sched = _make_scheduler()
        self.assertEqual(sched._get_sibling_numas_in_socket(0, exclude=0), [1])
        self.assertEqual(sched._get_sibling_numas_in_socket(0, exclude=1), [0])

    def test_acquire_one_cluster_prefers_affinity_numa(self):
        sched = _make_scheduler()
        self.assertEqual(sched._acquire_one_cluster(numa_id=0, socket_id=0), 0)

    def test_acquire_one_cluster_fallback_to_sibling_numa(self):
        sched = _make_scheduler()
        sched._used_clusters.update([0, 1])  # numa0 全部占满，降级到 numa1 的 cluster2
        self.assertEqual(sched._acquire_one_cluster(numa_id=0, socket_id=0), 2)

    def test_acquire_clusters_for_processes_fallback(self):
        sched = _make_scheduler()
        sched._used_clusters.update([0, 1])
        self.assertEqual(sched._acquire_clusters_for_processes(numa_id=0, socket_id=0), [2, 3])

    def test_acquire_clusters_for_processes_empty_when_exhausted(self):
        sched = _make_scheduler()
        sched._used_clusters.update([0, 1, 2, 3])
        self.assertEqual(sched._acquire_clusters_for_processes(numa_id=0, socket_id=0), [])


class TestAssignCpusToTasksColocated(unittest.TestCase):
    """UT for _assign_cpus_to_tasks (colocated 路径) 的 numa 元数据正确性

    cluster 可能降级到同 socket 的兄弟 NUMA，cpus 来自实际分配的 cluster 的核，
    task.numa/socket/cluster 由基类按 cpus 反查（get_*_of_cpus）得到，
    跟随 cpus 实际所属 numa，而非调用方传入的 NPU 亲和 NUMA。
    """

    def _sched_with_cpus(self):
        sched = _make_scheduler()
        sched.domain.get_cpus_of_clusters.side_effect = lambda clusters=None: list(range(8)) if clusters else []
        return sched

    def test_no_fallback_numa_matches_cluster(self):
        """非降级：cpus 取自 numa0 的 cluster0 核，task.numa=[0]。"""
        sched = self._sched_with_cpus()
        task = ProcessTask(group_id=0, pid=1000)
        sched._assign_cpus_to_tasks(cpus=[0, 1], tasks=[task])
        self.assertEqual(task.numa, [0])
        self.assertEqual(task.cluster, [0])

    def test_fallback_numa_follows_actual_cluster(self):
        """降级：cpus 取自兄弟 numa1 的 cluster2 核（4,5），
        task.numa 按 cpus 反查为 [1]。
        """
        sched = self._sched_with_cpus()
        task = ProcessTask(group_id=0, pid=1000)
        sched._assign_cpus_to_tasks(cpus=[4, 5], tasks=[task])
        # cpus[4,5] 属 cluster2/numa1，task.numa 跟随 cpus 实际所属 numa
        self.assertEqual(task.numa, [1])
        self.assertEqual(task.cluster, [2])

    def test_fallback_numa_consistent_for_memory_migration(self):
        """内存迁移目标 NUMA（process.numa[0]）应与 cpus 实际所属 NUMA 一致。"""
        sched = self._sched_with_cpus()
        task = ProcessTask(group_id=0, pid=1000)
        sched._assign_cpus_to_tasks(cpus=[6, 7], tasks=[task])
        self.assertEqual(task.numa[0], sched.domain.get_numas_of_clusters([3])[0])


class TestAssignClustersForBackgroundTasks(unittest.TestCase):
    """UT for _assign_clusters_for_background_tasks 的新策略。

    优先用空闲numa（socket无NPU亲和）的全部cpu，否则在非NPU亲和numa中预留20%cluster；
    复用基类 _do_assign_cpus_for_background_tasks 并回填 _numa_to_background_clusters/_used_clusters。
    """

    def _make_sched_with_npu_affinity(self, npu_to_numa):
        sched = _make_scheduler()
        sched._npu_to_numa = dict(npu_to_numa)
        numa_to_socket = {0: 0, 1: 0, 2: 1, 3: 1}
        sched._npu_to_socket = {npu: numa_to_socket[numa] for npu, numa in npu_to_numa.items()}
        return sched

    def test_unused_numas_when_socket_has_no_npu(self):
        sched = self._make_sched_with_npu_affinity({0: 0})  # NPU0 -> numa0(socket0)
        sched._assign_clusters_for_background_tasks()
        # socket1 无 NPU 亲和 -> unused_numas=[2,3]，base 用其全部 cpu
        self.assertEqual(sched.task.background_tasks_cpus, [8, 9, 10, 11, 12, 13, 14, 15])
        self.assertEqual(dict(sched._numa_to_background_clusters), {})
        self.assertEqual(sched._used_clusters, set())

    def test_reserve_when_all_sockets_have_npu(self):
        sched = self._make_sched_with_npu_affinity({0: 0, 1: 2})  # socket0/1 各一 NPU
        sched._assign_clusters_for_background_tasks()
        # to_reserve_numas=[1,3]；reserve_count=max(1,int(2*0.2))=1 -> numa1:[2], numa3:[6]
        self.assertEqual(dict(sched._numa_to_background_clusters), {1: [2], 3: [6]})
        self.assertEqual(sched._used_clusters, {2, 6})
        self.assertEqual(sched.task.background_tasks_cpus, [4, 5, 12, 13])

    def test_reserve_excludes_npu_affinity_numas(self):
        sched = self._make_sched_with_npu_affinity({0: 0, 1: 2})
        sched._assign_clusters_for_background_tasks()
        self.assertNotIn(0, sched._numa_to_background_clusters)
        self.assertNotIn(2, sched._numa_to_background_clusters)


class TestMultiNumaAffinityGroup(unittest.TestCase):
    """UT for 多NUMA亲和组（各进程绑定的NPU可亲和到不同NUMA，按各NPU自身亲和NUMA归一处理）。"""

    def _mk(self, npu_to_numa, bind_pids=None):
        bind_pids = bind_pids or {}
        sched = _make_scheduler()
        sched._npu_to_numa = dict(npu_to_numa)
        numa_to_socket = {0: 0, 1: 0, 2: 1, 3: 1}
        sched._npu_to_socket = {n: numa_to_socket[nid] for n, nid in npu_to_numa.items()}
        group = MagicMock()
        group.npu_tasks = {nid: MagicMock(**{"bind_pid": bind_pids.get(nid), "min_cpu": 1}) for nid in npu_to_numa}
        group.process_tasks, group.thread_tasks = {}, {}
        group.numa, group.socket, group.cluster, group.cpus = [], [], [], MagicMock()
        sched.task.get_group.return_value = group
        return sched, group

    def test_validate_allows_different_numas(self):
        # NPU0亲和numa0、NPU1亲和numa2（不同socket），校验不再raise
        sched, _ = self._mk({0: 0, 1: 2})
        sched._get_task_groups_to_schedule = lambda: [0]
        sched._validate_affinity_group_constraints()  # 不抛异常即通过

    def test_distribute_multi_socket(self):
        # NPU0亲和numa0(socket0)、NPU1亲和numa2(socket1)，组应记录两NUMA两socket
        sched, group = self._mk({0: 0, 1: 2})
        sched._get_task_groups_to_schedule = lambda: [0]
        sched._distribute_task_groups_to_sockets()
        self.assertEqual(sched._task_group_to_numas[0], [0, 2])
        self.assertEqual(sched._task_group_to_sockets[0], [0, 1])
        self.assertIn(0, sched._socket_to_task_groups)
        self.assertIn(1, sched._socket_to_task_groups)

    def test_isolated_bound_procs_use_per_npu_numa(self):
        # NPU0亲和numa0(cluster0,1)，绑定进程绑numa0其它cluster；不应绑到其它numa
        sched, group = self._mk({0: 0}, bind_pids={0: 1000})
        sched._task_group_to_numas = {0: [0]}
        sched._task_group_to_sockets = {0: [0]}
        b = MagicMock(**{"task_id": 1000})
        group.process_tasks = {1000: b}
        cl = sched._npu_process_cluster_mode_isolated(group_id=0, tasks=[b])
        self.assertEqual(set(cl), {0, 1})  # numa0可用cluster0,1

    def test_isolated_unbound_tasks_use_all_bind_clusters(self):
        # 两绑定进程各取numa0/numa2 cluster，未绑进程范围绑到并集
        sched, group = self._mk({0: 0, 1: 2}, bind_pids={0: 1000, 1: 2000})
        sched._task_group_to_numas = {0: [0, 2]}
        sched._task_group_to_sockets = {0: [0, 1]}
        b0 = MagicMock(**{"task_id": 1000})
        b1 = MagicMock(**{"task_id": 2000})
        u0 = MagicMock(**{"task_id": 9000})
        group.process_tasks = {1000: b0, 2000: b1, 9000: u0}
        cl = sched._npu_process_cluster_mode_isolated(group_id=0, tasks=[b0, b1, u0])
        self.assertEqual(set(cl), {0, 1, 4, 5})
        # 未绑进程绑到并集核(0-3,8-11)
        self.assertEqual(set(u0.set_cpu.call_args[0][0]), {0, 1, 2, 3, 8, 9, 10, 11})

    def test_colocated_high_prio_uses_per_npu_numa(self):
        # NPU0亲和numa0，高优先级线程选cluster0（numa0可用cluster0,1）
        sched, group = self._mk({0: 0})
        nt = group.npu_tasks[0]
        cl = sched._npu_process_cluster_mode_colocated_high_prio(group_id=0, npu_id=0, npu_task=nt)
        self.assertEqual(cl, 0)
        self.assertEqual(nt.cluster, [0])
        self.assertEqual(nt.numa, [0])

    def test_colocated_bound_proc_uses_remain_cpus_of_its_npu_cluster(self):
        # NPU0亲和numa0(cluster0)、NPU1亲和numa2(cluster4)，各绑定进程绑到各自cluster剩余核
        sched, group = self._mk({0: 0, 1: 2}, bind_pids={0: 1000, 1: 2000})
        b0 = MagicMock(**{"task_id": 1000})
        b1 = MagicMock(**{"task_id": 2000})
        group.process_tasks = {1000: b0, 2000: b1}
        npu_to_cluster = {0: 0, 1: 4}  # 高优先级线程已选cluster
        cl = sched._npu_process_cluster_mode_colocated(
            group_id=0,
            npu_bind_processes=[b0, b1],
            other_processes=[],
            other_threads=[],
            npu_to_cluster=npu_to_cluster,
        )
        self.assertEqual(set(cl), {0, 4})
        # cluster0 cpus[0,1]，min_cpu=1，剩余[1]；cluster4 cpus[8,9]，剩余[9]
        self.assertEqual(b0.set_cpu.call_args[0][0], [1])
        self.assertEqual(b1.set_cpu.call_args[0][0], [9])

    def test_colocated_unbound_tasks_share_all_remain_cpus(self):
        # 未绑进程共享所有NPU高优先级线程cluster的剩余核并集（共享池）
        sched, group = self._mk({0: 0, 1: 2}, bind_pids={0: 1000, 1: 2000})
        u0 = MagicMock(**{"task_id": 9000})
        group.process_tasks = {9000: u0}
        npu_to_cluster = {0: 0, 1: 4}
        cl = sched._npu_process_cluster_mode_colocated(
            group_id=0,
            npu_bind_processes=[],
            other_processes=[u0],
            other_threads=[],
            npu_to_cluster=npu_to_cluster,
        )
        self.assertEqual(set(cl), {0, 4})
        # 共享池：cluster0剩余[1] ∪ cluster4剩余[9] = {1,9}
        self.assertEqual(set(u0.set_cpu.call_args[0][0]), {1, 9})

    def test_shared_high_prio_per_numa_independent_decision(self):
        # NPU0亲和numa0(cluster0,1)、NPU1亲和numa2(cluster4,5)，各NUMA独立决策，各独占一cluster
        sched, group = self._mk({0: 0, 1: 2})
        res = sched._npu_process_cluster_mode_shared_high_prio(group_id=0)
        self.assertEqual(res, {0: 0, 1: 4})

    def test_shared_bound_procs_use_per_npu_numa(self):
        # NPU0亲和numa0、NPU1亲和numa2，各绑定进程绑到各自NUMA其它cluster
        # shared_high_prio已占用cluster0/4，绑定进程取各自NUMA剩余cluster
        sched, group = self._mk({0: 0, 1: 2}, bind_pids={0: 1000, 1: 2000})
        sched._task_group_to_numas = {0: [0, 2]}
        sched._task_group_to_sockets = {0: [0, 1]}
        sched._used_clusters = {0, 4}  # shared_high_prio 已占用
        b0 = MagicMock(**{"task_id": 1000})
        b1 = MagicMock(**{"task_id": 2000})
        group.process_tasks = {1000: b0, 2000: b1}
        cl = sched._npu_process_cluster_mode_shared(group_id=0, tasks=[b0, b1])
        # NPU0绑numa0剩余cluster(1)、NPU1绑numa2剩余cluster(5)
        self.assertEqual(set(cl), {1, 5})

    def test_shared_unbound_tasks_use_all_bind_clusters(self):
        # 两绑定进程各取numa0/numa2剩余cluster，未绑进程范围绑到并集（共享池）
        sched, group = self._mk({0: 0, 1: 2}, bind_pids={0: 1000, 1: 2000})
        sched._task_group_to_numas = {0: [0, 2]}
        sched._task_group_to_sockets = {0: [0, 1]}
        sched._used_clusters = {0, 4}  # shared_high_prio 已占用
        b0 = MagicMock(**{"task_id": 1000})
        b1 = MagicMock(**{"task_id": 2000})
        u0 = MagicMock(**{"task_id": 9000})
        group.process_tasks = {1000: b0, 2000: b1, 9000: u0}
        cl = sched._npu_process_cluster_mode_shared(group_id=0, tasks=[b0, b1, u0])
        self.assertEqual(set(cl), {1, 5})
        # 未绑进程绑到并集核：cluster1(cpus2,3) ∪ cluster5(cpus10,11)
        self.assertEqual(set(u0.set_cpu.call_args[0][0]), {2, 3, 10, 11})


class TestColocatedConfig(unittest.TestCase):
    """UT for colocated 布局常量与 config (commit ca71c77)"""

    def test_npu_process_cluster_mode_constant_value(self):
        self.assertEqual(cfg.NPU_PROCESS_CLUSTER_MODE_COLOCATED, "colocated")

    def test_set_colocated_npu_process_cluster_mode(self):
        c = AffinityConfig()
        c.set_schedule_strategy("cluster", npu_process_cluster_mode="colocated", enable_npu_topo_affinity=True)
        self.assertEqual(c.npu_process_cluster_mode, cfg.NPU_PROCESS_CLUSTER_MODE_COLOCATED)


class TestSharedConfig(unittest.TestCase):
    """UT for shared 布局常量与 config (commit 4c1f59c)"""

    def test_npu_process_cluster_mode_constant_value(self):
        self.assertEqual(cfg.NPU_PROCESS_CLUSTER_MODE_SHARED, "shared")

    def test_set_shared_npu_process_cluster_mode(self):
        c = AffinityConfig()
        c.set_schedule_strategy("cluster", npu_process_cluster_mode="shared", enable_npu_topo_affinity=True)
        self.assertEqual(c.npu_process_cluster_mode, cfg.NPU_PROCESS_CLUSTER_MODE_SHARED)

    def test_all_three_npu_process_cluster_modes_distinct(self):
        self.assertEqual(
            len(
                {
                    cfg.NPU_PROCESS_CLUSTER_MODE_ISOLATED,
                    cfg.NPU_PROCESS_CLUSTER_MODE_COLOCATED,
                    cfg.NPU_PROCESS_CLUSTER_MODE_SHARED,
                }
            ),
            3,
        )


class TestSharedCapacityArithmetic(unittest.TestCase):
    """UT for shared 的容量/分摊算术 (commit 4c1f59c)

    shared 策略2的核心是：cluster数 < NPU数时，求最小的每cluster容纳数k，
    使得 ceil(npu_num/k) <= cluster_num 且 k <= min(cluster_capacity)。
    这里对该算术本身（不依赖 domain）做断言，覆盖均衡分配的边界。
    """

    @staticmethod
    def _solve_k(npu_num: int, cluster_num: int, min_capacity: int) -> int:
        """复用生产侧 HbsNpuAffinityIsolateCluster._solve_min_k 求解 k，避免逻辑重复。"""
        return HbsNpuAffinityIsolateCluster._solve_min_k(npu_num, cluster_num, min_capacity)

    def test_clusters_ge_npus_k_is_1(self):
        """cluster数 >= NPU数，每NPU独占，k=1。"""
        self.assertEqual(self._solve_k(2, 4, 4), 1)
        self.assertEqual(self._solve_k(4, 4, 2), 1)

    def test_even_share_k(self):
        """4 NPU、2 cluster、每cluster可容2 -> k=2，均衡。"""
        self.assertEqual(self._solve_k(4, 2, 2), 2)

    def test_uneven_share_k(self):
        """7 NPU、3 cluster、每cluster可容3 -> k=3(ceil(7/3)=3<=3)。"""
        self.assertEqual(self._solve_k(7, 3, 3), 3)

    def test_k_capped_by_min_capacity(self):
        """min_capacity 不足以均衡时，k 退化为 min_capacity。"""
        # 5 NPU、2 cluster、min_capacity=1：k=1时 need=5>2；k 退化为1
        self.assertEqual(self._solve_k(5, 2, 1), 1)

    def test_ceil_distribution_no_loss(self):
        """用 k 计算的 need_clusters 可容纳所有 NPU（无遗漏）。"""
        for npu_num in range(1, 10):
            for cluster_num in range(1, 10):
                for min_capacity in range(1, 6):
                    k = self._solve_k(npu_num, cluster_num, min_capacity)
                    need = (npu_num + k - 1) // k
                    # need 不超过 cluster_num（或 k 已被 cap 到 min_capacity）
                    self.assertTrue(need <= cluster_num or k == min_capacity)


class TestSharedHighPrioStrategy1Exclusive(unittest.TestCase):
    """UT for _npu_process_cluster_mode_shared_high_prio 策略1：cluster数 >= NPU数，独占 (commit 4c1f59c)

    所有 NPU 亲和同一 NUMA（单组），按各自亲和NUMA归一处理。
    """

    def _make_sched_with_group(self, npu_ids, clusters_in_numa, numa_id=0):
        """构造带 group 的 scheduler，模拟策略1（cluster数>=NPU数）。

        clusters_in_numa: 该 numa 下可用的 cluster id 列表
        """
        sched = _make_scheduler()
        # 所有 NPU 亲和同一 numa/socket，按各NPU自身亲和NUMA归一处理
        sched._npu_to_numa = {nid: numa_id for nid in npu_ids}
        sched._npu_to_socket = {nid: 0 for nid in npu_ids}

        group = MagicMock()
        npu_tasks = {}
        for nid in npu_ids:
            nt = MagicMock()
            nt.min_cpu = 3
            npu_tasks[nid] = nt
        group.npu_tasks = npu_tasks
        group_id = 0
        sched.task.get_group.return_value = group

        # domain.get_cpus_of_clusters: 每个 cluster 返回 8 个核
        sched.domain.get_cpus_of_clusters.side_effect = lambda clusters=None: list(range(8)) if clusters else []

        # _acquire_clusters_for_processes 返回 numa 可用 cluster
        sched._acquire_clusters_for_processes = lambda numa_id, socket_id: list(clusters_in_numa)
        # _assign_shared_high_prio_cluster 记录到 npu_to_cluster 并标记 used
        assigned = {}

        def _assign(group_id, npu_id, npu_task, cluster, cpus=None):
            assigned[npu_id] = cluster
            sched._used_clusters.add(cluster)

        sched._assign_shared_high_prio_cluster = _assign
        return sched, group_id

    def test_strategy1_each_npu_exclusive_cluster(self):
        """4 NPU、4 cluster -> 每个 NPU 独占一个 cluster。"""
        sched, gid = self._make_sched_with_group([0, 1, 2, 3], [10, 11, 12, 13])
        res = sched._npu_process_cluster_mode_shared_high_prio(group_id=gid)
        self.assertEqual(res, {0: 10, 1: 11, 2: 12, 3: 13})

    def test_strategy1_more_clusters_than_npus(self):
        """2 NPU、4 cluster -> 各取前两个 cluster。"""
        sched, gid = self._make_sched_with_group([0, 1], [10, 11, 12, 13])
        res = sched._npu_process_cluster_mode_shared_high_prio(group_id=gid)
        self.assertEqual(res, {0: 10, 1: 11})

    def test_strategy3_insufficient_capacity_raises(self):
        """cluster 总容量不足以容纳所有 NPU 时抛 RuntimeError（策略3）。"""
        sched, gid = self._make_sched_with_group([0, 1, 2, 3], [10, 11])
        # 每个 cluster 8 核、每 NPU 需 5 核 -> 每 cluster 容量 8//5=1，2 cluster 总容量 2 < 4 NPU
        for nt in sched.task.get_group.return_value.npu_tasks.values():
            nt.min_cpu = 5
        with self.assertRaises(RuntimeError):
            sched._npu_process_cluster_mode_shared_high_prio(group_id=gid)

    def test_no_available_cluster_raises(self):
        """无可用 cluster 时抛 RuntimeError。"""
        sched, gid = self._make_sched_with_group([0, 1], [])
        with self.assertRaises(RuntimeError):
            sched._npu_process_cluster_mode_shared_high_prio(group_id=gid)


if __name__ == "__main__":
    unittest.main()

