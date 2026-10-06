import unittest
from unittest.mock import Mock, patch

from a_sched.backup import AffinityBackup
from a_sched.engine import AffinityEngine


class TestNumaBalancing(unittest.TestCase):
    def setUp(self):
        self.backup = AffinityBackup(task=Mock(), domain=Mock(), cpuset=Mock())

    @patch("a_sched.backup.utils.execute_command", return_value=("1\n", "", 0))
    def test_backup_detects_enabled_numa_balancing(self, execute_command):
        self.assertTrue(self.backup.build_numa_balancing_status())
        execute_command.assert_called_once_with(["sysctl", "-n", "kernel.numa_balancing"])

    @patch("a_sched.backup.utils.execute_command", return_value=("", "", 0))
    def test_restore_enables_numa_balancing(self, execute_command):
        self.backup.restore_numa_balancing_status(True)
        execute_command.assert_called_once_with(["sysctl", "-w", "kernel.numa_balancing=1"])

    @patch("a_sched.engine.utils.execute_command", return_value=("", "", 0))
    def test_engine_disables_numa_balancing(self, execute_command):
        engine = AffinityEngine()
        engine._stop_numa_balancing()
        execute_command.assert_called_once_with(["sysctl", "-w", "kernel.numa_balancing=0"])


if __name__ == "__main__":
    unittest.main()
