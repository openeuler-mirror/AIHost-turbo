import unittest
from unittest.mock import Mock, patch

from a_sched.engine import AffinityEngine
from a_sched.task.npu_task import NpuTaskA3
from a_sched import utils


class TestProcessRuntimeControls(unittest.TestCase):
    @patch("a_sched.utils.psutil.process_iter")
    def test_get_pid_by_process_name_filters_child_processes(self, process_iter):
        process_iter.return_value = [
            Mock(info={"pid": 100, "name": "xllm", "ppid": 1}),
            Mock(info={"pid": 101, "name": "xllm", "ppid": 100}),
            Mock(info={"pid": 200, "name": "other", "ppid": 1}),
        ]

        self.assertEqual(utils.get_pid_by_process_name("xllm", top_level=True), [(100, "xllm")])

    @patch("a_sched.task.npu_task.utils.get_pid_by_process_name")
    @patch("a_sched.task.npu_task.os.access", return_value=False)
    def test_npu_dev_sq_scan_skips_without_write_access(self, _access, get_processes):
        NpuTaskA3(group_id=0, npu_id=0)._get_npu_dev_sq()
        get_processes.assert_not_called()

    @patch("a_sched.task.npu_task.utils.get_npu_irq_by_name")
    @patch("a_sched.task.npu_task.os.access", return_value=False)
    def test_npu_irq_scan_skips_without_write_access(self, _access, get_irqs):
        NpuTaskA3(group_id=0, npu_id=0)._get_npu_irqs()
        get_irqs.assert_not_called()

    @patch("a_sched.engine.utils.migrate_process_pages")
    def test_memory_migration_runs_for_configured_process(self, migrate_pages):
        process = Mock(task_id=123, numa=[1])
        group = Mock(process_tasks={123: process})
        engine = AffinityEngine()
        engine.task.groups = {0: group}
        engine.domain.get_all_numas_id = Mock(return_value=[0, 1])

        engine._bind_memory()

        migrate_pages.assert_called_once_with(pid=123, src_numa=[0, 1], tgt_numa=1)


if __name__ == "__main__":
    unittest.main()
