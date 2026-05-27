from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field
import os
import re

from a_sched.utils import CPUMask
from a_sched.config import AffinityConfig
import a_sched.utils as utils


class DomainLevel(Enum):
    """亲和域层级（模仿调度器层级）"""

    SOCKET = 0  # 最高层：物理CPU插槽
    NUMA = 1  # 中间层：NUMA内存节点
    CLUSTER = 2  # 最底层：CPU性能集群
    THREAD = 3  # 额外：线程级（可选）


class AffinityDomain(ABC):
    """
    亲和域抽象基类
    模仿Linux调度器sched_domain的设计
    """

    def __init__(self, level: DomainLevel, domain_id: int):
        self.level = level
        self.domain_id = domain_id

        # 核心拓扑属性（模仿sched_domain）
        self.cpus = CPUMask()  # 该域包含的CPU位图
        self.parent: AffinityDomain | None = None  # 父域
        self.children: list[AffinityDomain] = []  # 子域

        # 层级名称
        self._level_names = {
            DomainLevel.SOCKET: "SOCKET",
            DomainLevel.NUMA: "NUMA",
            DomainLevel.CLUSTER: "CLUSTER",
            DomainLevel.THREAD: "THREAD",
        }

    @abstractmethod
    def detect_from_hardware(self) -> bool:
        """从硬件检测域信息"""
        pass

    def add_child(self, child: AffinityDomain):
        child.parent = self
        self.children.append(child)

    def get_all_children(self) -> list[AffinityDomain]:
        return self.children

    def get_all_children_id(self) -> list[int]:
        children_id = []
        for child in self.children:
            children_id.append(child.domain_id)
        return children_id

    def get_children_num(self) -> int:
        return len(self.children)

    def __str__(self) -> str:
        level_name = self._level_names.get(self.level, "UNKNOWN")
        return f"{level_name}[{self.domain_id}]: CPUs={self.cpus}"


class SocketDomain(AffinityDomain):
    """Socket域 - 物理CPU插槽（最高层）"""

    def __init__(self, socket_id: int):
        super().__init__(level=DomainLevel.SOCKET, domain_id=socket_id)

    def detect_from_hardware(self) -> bool:
        return True


class NumaDomain(AffinityDomain):
    """NUMA域 - 内存节点（中间层）"""

    def __init__(self, node_id: int):
        super().__init__(level=DomainLevel.NUMA, domain_id=node_id)

    def detect_from_hardware(self) -> bool:
        return True


class ClusterDomain(AffinityDomain):
    """Cluster域 - CPU性能集群（最底层）"""

    def __init__(self, cluster_id: int):
        super().__init__(level=DomainLevel.CLUSTER, domain_id=cluster_id)

    def detect_from_hardware(self) -> bool:
        return True


@dataclass
class CpuCore:
    physical_package_id: int = -1
    cluster_id: int = -1


@dataclass
class NumaNode:
    cpulist: list[int] = field(default_factory=list)


class AffinityDomainBuilder:
    """
    亲和域构建
    """

    def __init__(self, config: AffinityConfig) -> None:
        self.socket_domains: list[SocketDomain] = []
        self.numa_domains: list[NumaDomain] = []
        self.cluster_domains: list[ClusterDomain] = []

        self._exclude_cpus = config.exclude_cpus
        self._container_cpus = utils.get_allowed_cpu_list()

        self._cpu_dict: dict[int, CpuCore] = {}
        self._numa_dict: dict[int, NumaNode] = {}

        self._socket_to_cpus: dict[int, list[int]] = defaultdict(list)
        self._numa_to_cpus: dict[int, list[int]] = defaultdict(list)
        self._cluster_to_cpus: dict[int, list[int]] = defaultdict(list)
        self._socket_to_numas: dict[int, set[int]] = defaultdict(set)
        self._numa_to_clusters: dict[int, set[int]] = defaultdict(set)

    def build_affinity_domain(self) -> bool:
        try:
            self._read_cpu()
            self._read_numa()
            self._build_topo()
            self._build_socket_domains()
            for socket in self.socket_domains:
                self._build_numa_domains_for_socket(socket)
            for numa in self.numa_domains:
                self._build_cluster_domains_for_numa(numa)
            return True

        except Exception as e:
            print(f"Build affinity domain fail: {e}.")
            return False

    def _read_cpu(self) -> None:
        cpu_root_dir = "/sys/devices/system/cpu/"
        cpu_pattern = re.compile(r"^cpu(\d+)$")

        for entry in utils.safe_listdir(cpu_root_dir):
            match = cpu_pattern.match(entry)
            if not match:
                continue
            cpu_id = int(match.group(1))

            # 过滤掉不可用的cpu
            if not self._is_cpu_available(cpu_id):
                continue

            # 过滤掉不在线的cpu
            if not utils.is_cpu_online(cpu_id):
                continue

            cpu_dir = os.path.join(cpu_root_dir, f"cpu{cpu_id}")
            topology_dir = os.path.join(cpu_dir, "topology/")

            cpu_core = CpuCore(
                physical_package_id=utils.read_int_param(topology_dir, "physical_package_id"),
                cluster_id=utils.read_int_param(topology_dir, "cluster_id"),
            )
            self._cpu_dict[cpu_id] = cpu_core

    def _read_numa(self) -> None:
        node_root_dir = "/sys/devices/system/node/"
        node_pattern = re.compile(r"^node(\d+)$")

        for entry in utils.safe_listdir(node_root_dir):
            match = node_pattern.match(entry)
            if not match:
                continue
            node_id = int(match.group(1))
            node_dir = os.path.join(node_root_dir, f"node{node_id}")
            numa = NumaNode(cpulist=utils.read_list_param(node_dir, "cpulist")[0])
            self._numa_dict[node_id] = numa

    def _is_cpu_available(self, cpu: int) -> bool:
        """
        cpu是否可用
            1) 容器场景如果不在容器绑定的cpu范围需要剔除
            2) 在通过set_exclude_cpu设置的排除范围需要剔除
        """

        if self._container_cpus:
            if cpu not in self._container_cpus:
                return False
        if self._exclude_cpus:
            if cpu in self._exclude_cpus:
                return False
        return True

    def _build_topo(self) -> None:
        for cpu_id, cpu_core in self._cpu_dict.items():
            if cpu_core.physical_package_id != -1:
                self._socket_to_cpus[cpu_core.physical_package_id].append(cpu_id)
            # 部分硬件不存在cluster，按照cluster等于cpu进行处理，保持cluster级抽象
            if cpu_core.cluster_id == -1:
                cpu_core.cluster_id = cpu_id
            self._cluster_to_cpus[cpu_core.cluster_id].append(cpu_id)

        for numa_id, numa in self._numa_dict.items():
            for cpu_id in numa.cpulist:
                cpu = self._cpu_dict.get(cpu_id)
                if cpu is None:
                    continue
                self._numa_to_cpus[numa_id].append(cpu_id)
                if cpu.physical_package_id != -1:
                    self._socket_to_numas[cpu.physical_package_id].add(numa_id)
                if cpu.cluster_id != -1:
                    self._numa_to_clusters[numa_id].add(cpu.cluster_id)

    def _build_socket_domains(self) -> None:
        for socket_id, cpu_list in sorted(self._socket_to_cpus.items()):
            socket = SocketDomain(socket_id)
            socket.detect_from_hardware()
            socket.cpus.from_list(cpu_list)
            self.socket_domains.append(socket)

    def _build_numa_domains_for_socket(self, socket: SocketDomain) -> None:
        socket_numas = self._socket_to_numas.get(socket.domain_id)
        if socket_numas is None:
            return
        for numa_id in sorted(socket_numas):
            numa = NumaDomain(numa_id)
            numa.detect_from_hardware()
            cpu_list = self._numa_to_cpus.get(numa_id, [])
            numa.cpus.from_list(cpu_list)
            self.numa_domains.append(numa)
            socket.add_child(numa)

    def _build_cluster_domains_for_numa(self, numa: NumaDomain) -> None:
        numa_clusters = self._numa_to_clusters.get(numa.domain_id)
        if numa_clusters is None:
            return
        for cluster_id in sorted(numa_clusters):
            cluster = ClusterDomain(cluster_id)
            cluster.detect_from_hardware()
            cpu_list = self._cluster_to_cpus.get(cluster_id, [])
            cluster.cpus.from_list(cpu_list)
            self.cluster_domains.append(cluster)
            numa.add_child(cluster)


class AffinityDomainManager:
    """
    亲和域管理
    """

    def __init__(self, config: AffinityConfig) -> None:
        self.socket_domains: list[SocketDomain] = []
        self.numa_domains: list[NumaDomain] = []
        self.cluster_domains: list[ClusterDomain] = []
        self._config = config

    def build_affinity_domain(self) -> None:
        builder = AffinityDomainBuilder(self._config)
        if builder.build_affinity_domain():
            self.socket_domains = builder.socket_domains
            self.numa_domains = builder.numa_domains
            self.cluster_domains = builder.cluster_domains

    def get_socket_num(self) -> int:
        return len(self.socket_domains)

    def get_socket_domain(self, socket_id: int) -> SocketDomain | None:
        for socket in self.socket_domains:
            if socket.domain_id == socket_id:
                return socket
        return None

    def get_numa_domain(self, numa_id: int) -> NumaDomain | None:
        for numa in self.numa_domains:
            if numa.domain_id == numa_id:
                return numa
        return None

    def get_all_numas_id(self) -> list:
        numa_list: list = []
        for numa in self.numa_domains:
            numa_list.append(numa.domain_id)
        return numa_list

    def get_core_num(self) -> int:
        core_num: int = 0
        for socket in self.socket_domains:
            core_num += socket.cpus.count()
        return core_num

    def get_cluster_domain(self, cluster_id: int) -> ClusterDomain | None:
        for cluster in self.cluster_domains:
            if cluster.domain_id == cluster_id:
                return cluster
        return None

    def print_all(self) -> None:
        print(
            f"TOTAL: sockets={self.get_socket_num()}, "
            f"numas={len(self.numa_domains)}, "
            f"clusters={len(self.cluster_domains)}, "
            f"cores={self.get_core_num()}"
        )
        for socket in self.socket_domains:
            self.print_domain(socket)

    def print_domain(self, domain: AffinityDomain, indent: int = 0) -> None:
        prefix = "  " * indent
        print(f"{prefix}{domain}")

        # 打印子域
        for child in domain.get_all_children():
            self.print_domain(child, indent + 1)

    def get_sockets_of_cpus(self, cpus: list[int]) -> list:
        socket_list: list[int] = []
        for socket in self.socket_domains:
            for cpu in cpus:
                if socket.cpus.isset(cpu=cpu):
                    socket_list.append(socket.domain_id)
                    break
        return sorted(set(socket_list))

    def get_numas_of_cpus(self, cpus: list[int]) -> list:
        numa_list: list[int] = []
        for numa in self.numa_domains:
            for cpu in cpus:
                if numa.cpus.isset(cpu=cpu):
                    numa_list.append(numa.domain_id)
                    break
        return sorted(set(numa_list))

    def get_clusters_of_cpus(self, cpus: list[int]) -> list:
        cluster_list: list[int] = []
        for cluster in self.cluster_domains:
            for cpu in cpus:
                if cluster.cpus.isset(cpu=cpu):
                    cluster_list.append(cluster.domain_id)
                    break
        return sorted(set(cluster_list))
