import unittest
from unittest.mock import MagicMock, patch

from a_sched import utils
from a_sched.task.task_manager import TaskManager
from a_sched.utils import AscendDeviceType


def create_task_manager():
    return TaskManager(domain=None)


class TestAutoBindNpuForProcess(unittest.TestCase):
    def test_a2_skips_npu_process_lookup(self):
        manager = create_task_manager()
        group_id = manager.group_create(name="group")
        with (
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A2),
            patch.object(utils, "get_npu_topo_process_id") as get_process_ids,
        ):
            manager._auto_bind_npu_for_process(group_id=group_id, pid=100)

        get_process_ids.assert_not_called()
        self.assertIsNone(manager._pid_to_npu)
        self.assertEqual(manager.process_to_npu, {})

    def test_unmapped_process_is_not_bound(self):
        manager = create_task_manager()
        group_id = manager.group_create(name="group")
        with (
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(utils, "get_npu_topo_process_id", return_value={0: 1000}),
        ):
            manager._auto_bind_npu_for_process(group_id=group_id, pid=2000)

        self.assertEqual(manager.process_to_npu, {})
        self.assertEqual(manager._pid_to_npu, {1000: 0})

    def test_npu_process_is_bound_automatically(self):
        manager = create_task_manager()
        group_id = manager.group_create(name="group")
        manager.groups[group_id].process_tasks[1000] = MagicMock()
        with (
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(utils, "get_npu_topo_process_id", return_value={0: 1000}),
        ):
            manager._auto_bind_npu_for_process(group_id=group_id, pid=1000)

        self.assertEqual(manager.process_to_npu, {1000: 0})
        self.assertIn(0, manager.groups[group_id].npu_tasks)

    def test_process_mapping_is_cached(self):
        manager = create_task_manager()
        group_id = manager.group_create(name="group")
        with (
            patch.object(utils, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(utils, "get_npu_topo_process_id", return_value={0: 1000}) as get_process_ids,
        ):
            manager._auto_bind_npu_for_process(group_id=group_id, pid=2000)
            manager._auto_bind_npu_for_process(group_id=group_id, pid=3000)

        get_process_ids.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
