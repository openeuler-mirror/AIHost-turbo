import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules["psutil"] = MagicMock()
sys.modules["psutil"].__version__ = "0.0.0"

from a_sched import config as cfg  # noqa: E402
from a_sched.config import AffinityConfig  # noqa: E402
from a_sched.strategy.hbs_npu_affinity_isolate_cluster import HbsNpuAffinityIsolateCluster  # noqa: E402
from a_sched.strategy import scheduler_factory as sf  # noqa: E402
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
    from collections import defaultdict

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


if __name__ == "__main__":
    unittest.main()

