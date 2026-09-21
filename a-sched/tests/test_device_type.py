import subprocess
import unittest
from unittest.mock import patch

import a_sched.utils as utils


class TestAscendDeviceType(unittest.TestCase):
    def tearDown(self):
        utils._ascend_device_type = None

    @patch("a_sched.utils.subprocess.check_output")
    def test_detect_a3_queries_chip_zero(self, check_output):
        check_output.side_effect = [
            b"NPU ID : 0\n",
            b"NPU Name : Ascend\n",
            b"Chip Name : Ascend910B\nNPU Name : Ascend\n",
        ]

        self.assertEqual(utils.detect_ascend_device_type(), utils.AscendDeviceType.A3)
        check_output.assert_called_with(
            ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]
        )

    @patch("a_sched.utils.subprocess.check_output")
    def test_detect_a5_uses_board_info_directly(self, check_output):
        check_output.side_effect = [
            b"NPU ID : 1\n",
            b"Chip Name : Ascend950\nNPU Name : Ascend\n",
        ]

        self.assertEqual(utils.detect_ascend_device_type(), utils.AscendDeviceType.A5)
        self.assertEqual(check_output.call_count, 2)

    @patch("a_sched.utils.subprocess.check_output", side_effect=FileNotFoundError)
    def test_missing_npu_smi_returns_unknown(self, _check_output):
        self.assertEqual(utils.detect_ascend_device_type(), utils.AscendDeviceType.UNKNOWN)

    @patch("a_sched.utils.subprocess.check_output")
    def test_command_failure_is_reported(self, check_output):
        check_output.side_effect = subprocess.CalledProcessError(1, ["npu-smi"])

        with self.assertRaises(RuntimeError):
            utils.detect_ascend_device_type()


if __name__ == "__main__":
    unittest.main()
