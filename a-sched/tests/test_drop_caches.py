import unittest
from unittest.mock import Mock, mock_open, patch

import a_sched
from a_sched.config import AffinityConfig
from a_sched.engine import AffinityEngine
from a_sched.utils import drop_caches


class TestDropCaches(unittest.TestCase):
    def test_config_default_is_false(self):
        self.assertFalse(AffinityConfig().drop_caches)

    def test_public_api_updates_engine_config(self):
        a_sched.set_drop_caches(True)
        self.assertTrue(a_sched.api.affinity.config.drop_caches)
        a_sched.set_drop_caches(False)

    def test_drop_caches_is_public(self):
        self.assertIn("set_drop_caches", a_sched.__all__)
        self.assertTrue(callable(a_sched.set_drop_caches))

    def test_drop_caches_writes_kernel_control(self):
        opened = mock_open()
        with patch("builtins.open", opened):
            self.assertTrue(drop_caches())
        opened.assert_called_once_with("/proc/sys/vm/drop_caches", "w")
        opened().write.assert_called_once_with("1")

    def test_drop_caches_handles_permission_error(self):
        opened = mock_open()
        opened.side_effect = PermissionError("permission denied")
        with patch("builtins.open", opened):
            self.assertFalse(drop_caches())

    @patch("a_sched.engine.utils.drop_caches")
    def test_engine_drops_caches_before_memory_migration(self, drop_cache_pages):
        engine = AffinityEngine()
        engine.config.drop_caches = True
        engine.backup_affinity = Mock()
        engine._stop_irq_balance = Mock()
        engine._stop_numa_balancing = Mock()
        engine._bind_cpus = Mock()
        engine._bind_memory = Mock()

        engine._execute()

        drop_cache_pages.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
