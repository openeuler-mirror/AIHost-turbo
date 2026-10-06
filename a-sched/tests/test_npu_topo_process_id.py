import unittest
from unittest.mock import patch

from a_sched import utils
from a_sched.utils import AscendDeviceType


A5_INFO_OUTPUT = """\
+---------------------------+---------------+----------------------------------------------------------------------+-----------------------+
| NPU ID                    | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+=======================+
| 0                         | 107303        | python             | 39852                 | NA                      |
| 1                         | 107304        | python             | 39912                 | NA                      |
| 2                         | NA            | NA                 | NA                    | NA                      |
+---------------------------+---------------+----------------------------------------------------------------------+-----------------------+
"""

A3_INFO_OUTPUT = """\
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+========================================================+
| 0       0                 | 1803913       | VLLMWorker_DP            | 59887                   |
| 0       1                 | 1803992       | VLLMWorker_DP            | 60166                   |
| 1       0                 | 1803933       | VLLMWorker_DP            | 59887                   |
| 1       1                 | 1803959       | VLLMWorker_DP            | 60166                   |
+---------------------------+---------------+--------------------------------------------------------+
"""


class TestNpuTopoProcessId(unittest.TestCase):
    def test_parse_a5_process_table(self):
        self.assertEqual(utils._parse_npu_info_output_a5(A5_INFO_OUTPUT), {0: 107303, 1: 107304})

    def test_parse_a3_process_table(self):
        self.assertEqual(
            utils._parse_npu_info_output_a3(A3_INFO_OUTPUT),
            {0: 1803913, 1: 1803992, 2: 1803933, 3: 1803959},
        )

    def test_missing_process_table_returns_empty(self):
        self.assertEqual(utils._parse_npu_info_output_a5("missing table"), {})
        self.assertEqual(utils._parse_npu_info_output_a3("missing table"), {})

    def test_get_process_ids_for_a5(self):
        with (
            patch.object(utils, "execute_command", return_value=(A5_INFO_OUTPUT, "", 0)),
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        ):
            self.assertEqual(utils.get_npu_topo_process_id(), {0: 107303, 1: 107304})

    def test_get_process_ids_for_a3(self):
        with (
            patch.object(utils, "execute_command", return_value=(A3_INFO_OUTPUT, "", 0)),
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A3),
        ):
            self.assertEqual(
                utils.get_npu_topo_process_id(),
                {0: 1803913, 1: 1803992, 2: 1803933, 3: 1803959},
            )

    def test_unsupported_device_returns_empty(self):
        with (
            patch.object(utils, "execute_command", return_value=("output", "", 0)),
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A2),
        ):
            self.assertEqual(utils.get_npu_topo_process_id(), {})

    def test_command_failure_returns_empty(self):
        with patch.object(utils, "execute_command", return_value=("", "error", 1)):
            self.assertEqual(utils.get_npu_topo_process_id(), {})


if __name__ == "__main__":
    unittest.main()
