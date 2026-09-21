import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from a_sched.affinity_domain import AffinityDomainBuilder
from a_sched.config import AffinityConfig
from a_sched.task import NpuTaskA3


class TestSmtTopology(unittest.TestCase):
    @patch("a_sched.affinity_domain.utils.read_int_param")
    @patch("a_sched.affinity_domain.utils.is_cpu_online", return_value=True)
    @patch("a_sched.affinity_domain.utils.safe_listdir", return_value=["cpu0", "cpu4", "cpu1"])
    @patch("a_sched.affinity_domain.utils.get_allowed_cpu_list", return_value=[])
    def test_smt_siblings_are_deduplicated(
        self, _allowed_cpus, _listdir, _is_online, read_int_param
    ):
        core_ids = {0: 0, 1: 1, 4: 0}

        def read_topology_value(path, name):
            cpu_id = int(path.rstrip("/").split("cpu")[-1].split("/")[0])
            if name == "core_id":
                return core_ids[cpu_id]
            return 0

        read_int_param.side_effect = read_topology_value
        builder = AffinityDomainBuilder(AffinityConfig())

        builder._read_cpu()

        self.assertEqual(len(builder._cpu_dict), 2)
        self.assertEqual(set(builder._cpu_dict).intersection({0, 4}), {0})

    def test_release_thread_is_included_in_npu_plan(self):
        task = NpuTaskA3(group_id=0, npu_id=0)
        task._release_thread = 123
        task._release_thread_cpus = [2]

        output = io.StringIO()
        with redirect_stdout(output):
            task.print()

        self.assertIn("Thread[123](release_thread): cpu=2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
