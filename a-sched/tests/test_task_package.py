import unittest
from unittest.mock import patch

import a_sched
from a_sched.affinity_domain import AffinityDomainManager
from a_sched.config import AffinityConfig
from a_sched.task import (
    NpuTaskA3,
    NpuTaskA5,
    PriorityLevel,
    TaskGroup,
    TaskManager,
    ThreadTask,
    create_npu_task,
)
from a_sched.utils import AscendDeviceType


class TestTaskPackage(unittest.TestCase):
    def test_cpuset_api_remains_public(self):
        self.assertTrue(callable(a_sched.enable_cpuset_isolate))

    def test_task_manager_accepts_affinity_domain(self):
        domain = AffinityDomainManager(config=AffinityConfig())

        manager = TaskManager(domain=domain)

        self.assertIs(manager._domain, domain)

    @patch("a_sched.task.npu_task.utils.get_ascend_device_type")
    def test_create_npu_task_by_device_type(self, get_device_type):
        get_device_type.return_value = AscendDeviceType.A3
        self.assertIsInstance(create_npu_task(group_id=0, npu_id=0), NpuTaskA3)

        get_device_type.return_value = AscendDeviceType.A5
        self.assertIsInstance(create_npu_task(group_id=0, npu_id=0), NpuTaskA5)

    @patch("a_sched.task.task_group.utils.get_thread_pid_by_tid", return_value=100)
    def test_high_priority_threads_remain_available_for_cpuset(self, _get_pid):
        group = TaskGroup(group_id=0)
        thread = ThreadTask(group_id=0, tid=101, pid=100, name="worker")
        thread.priority = PriorityLevel.HIGH
        group.thread_tasks[thread.task_id] = thread

        self.assertEqual(group.get_high_prio_threads(), {100: [101]})


if __name__ == "__main__":
    unittest.main()
