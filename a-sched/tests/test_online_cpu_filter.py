import unittest
from unittest.mock import mock_open, patch

from a_sched.affinity_domain import AffinityDomainBuilder
from a_sched.config import AffinityConfig
from a_sched import utils


class TestOnlineCpuFilter(unittest.TestCase):
    def test_get_online_cpus_parses_kernel_range(self):
        with patch("builtins.open", mock_open(read_data="0-2,4\n")):
            self.assertEqual(utils.get_online_cpus(), {0, 1, 2, 4})

    @patch("a_sched.affinity_domain.utils.get_online_cpus", return_value={0, 2})
    @patch("a_sched.affinity_domain.utils.get_allowed_cpu_list", return_value=[])
    def test_domain_rejects_offline_cpu(self, _allowed_cpus, _online_cpus):
        builder = AffinityDomainBuilder(config=AffinityConfig())
        self.assertTrue(builder._is_cpu_available(0))
        self.assertFalse(builder._is_cpu_available(1))


if __name__ == "__main__":
    unittest.main()
