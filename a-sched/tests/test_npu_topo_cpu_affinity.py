import unittest
from unittest.mock import patch

from a_sched import utils
from a_sched.utils import AscendDeviceType


A5_TOPO_OUTPUT = """\
NPU0       NPU1       NPU2       NPU3       CPU Affinity
NPU0       X          UB         UB         144-191
NPU1       UB         X          UB         144-191
NPU2       UB         UB         X          48-95
NPU3       UB         UB         UB         X
"""


class TestNpuTopoCpuAffinity(unittest.TestCase):
    def test_parse_a5_topology(self):
        affinity = utils._parse_npu_topo_output_a5(A5_TOPO_OUTPUT)
        self.assertEqual(affinity[0], list(range(144, 192)))
        self.assertEqual(affinity[1], list(range(144, 192)))
        self.assertEqual(affinity[2], list(range(48, 96)))
        self.assertNotIn(3, affinity)

    def test_missing_affinity_header_returns_empty(self):
        self.assertEqual(utils._parse_npu_topo_output_a5("NPU0 NPU1\nNPU0 X\n"), {})

    def test_short_output_returns_empty(self):
        self.assertEqual(utils._parse_npu_topo_output_a5("only one line"), {})

    def test_non_a5_device_is_rejected(self):
        with (
            patch.object(utils, "execute_command", return_value=(A5_TOPO_OUTPUT, "", 0)),
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A3),
        ):
            self.assertEqual(utils.get_npu_topo_cpu_affinity(), {})

    def test_a5_topology_command_is_parsed(self):
        with (
            patch.object(utils, "execute_command", return_value=(A5_TOPO_OUTPUT, "", 0)),
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        ):
            affinity = utils.get_npu_topo_cpu_affinity()

        self.assertEqual(affinity[2], list(range(48, 96)))


if __name__ == "__main__":
    unittest.main()
