from __future__ import annotations
import os
import re
import psutil
import subprocess


class CPUMask:
    """
    基于 bytearray 实现的 CPU Mask 类
    核心特性：
    - 固定支持 0~1023 共 1024 个逻辑 CPU 核心
    - 用 bytearray 存储位图 (128 字节 = 1024 位)
    - 封装所有核心位操作，兼容 Linux CPU 掩码语义
    """

    # 固定常量（与 Linux cpu_set_t 对齐）
    MAX_CPUS = 1024  # 最大支持的逻辑 CPU 核心数
    BYTE_SIZE = MAX_CPUS // 8  # 所需字节数：1024 / 8 = 128

    def __init__(self):
        self._bitmap = bytearray(self.BYTE_SIZE)

    def zero(self) -> None:
        self._bitmap = bytearray(self.BYTE_SIZE)

    def set(self, cpu: int) -> None:
        if not (0 <= cpu < self.MAX_CPUS):
            raise ValueError(f"cpu must in range 0~{self.MAX_CPUS-1} (current:{cpu})")
        byte_idx = cpu // 8
        bit_idx = cpu % 8
        self._bitmap[byte_idx] |= 1 << bit_idx

    def clr(self, cpu: int) -> None:
        if not (0 <= cpu < self.MAX_CPUS):
            raise ValueError(f"cpu must in range 0~{self.MAX_CPUS-1} (current:{cpu})")
        byte_idx = cpu // 8
        bit_idx = cpu % 8
        self._bitmap[byte_idx] &= ~(1 << bit_idx)

    def isset(self, cpu: int) -> bool:
        if not (0 <= cpu < self.MAX_CPUS):
            return False
        byte_idx = cpu // 8
        bit_idx = cpu % 8
        return (self._bitmap[byte_idx] & (1 << bit_idx)) != 0

    def count(self) -> int:
        return sum(bin(byte).count("1") for byte in self._bitmap)

    def from_list(self, cpus: list[int]) -> CPUMask:
        self.zero()  # 先清空
        for cpu in cpus:
            self.set(cpu)
        return self

    def to_list(self) -> list[int]:
        enabled_cpus = []
        for cpu in range(self.MAX_CPUS):
            if self.isset(cpu):
                enabled_cpus.append(cpu)
        return enabled_cpus

    def from_mask(self, cpu_mask_str: str) -> CPUMask:
        hex_str = cpu_mask_str.replace(",", "").strip()
        hex_str_len = len(hex_str)
        hex_str = hex_str.zfill(hex_str_len if hex_str_len % 2 == 0 else (hex_str_len + 1))
        raw_bytes = bytes.fromhex(hex_str)[::-1]
        for byte_idx, byte_val in enumerate(raw_bytes):
            for bit in range(8):
                if byte_val & (1 << bit):
                    cpu = byte_idx * 8 + bit
                    self.set(cpu)
        return self

    def to_mask(self) -> str:
        bitmap_rev = self._bitmap[::-1]
        return bitmap_rev.hex(sep=",", bytes_per_sep=4).lstrip("0,")

    def __repr__(self) -> str:
        enabled = self.to_list()
        return f"CPUMask(enabled_cpus={enabled}, count={self.count()}, " f"hex_mask=0x{self.to_mask()})"

    def __str__(self) -> str:
        return compress_continuous(self.to_list())

    def __contains__(self, cpu: int) -> bool:
        return self.isset(cpu)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CPUMask):
            return False
        return self._bitmap == other._bitmap

    def set_range(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError(f"start {start} must LE than end {end}")
        for cpu in range(start, end + 1):
            self.set(cpu)

    def clr_range(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError(f"start {start} must LE than end {end}")
        for cpu in range(start, end + 1):
            self.clr(cpu)

    def set_list(self, cpu_list: list) -> None:
        for cpu in cpu_list:
            self.set(cpu)


def parse_cpu_affinity_string(affinity_str) -> list:
    """
    解析 CPU 亲和性字符串，返回绑定的 CPU 核心列表
    :param affinity_str: CPU 亲和字符串（如 "0-3,5"、"all"）
    :return: 有序、去重的 CPU 核心列表（int 类型）；解析失败返回空列表
    """
    cpu_list = []
    max_cpu_num = CPUMask.MAX_CPUS
    total_cpus = os.cpu_count() or 1  # 获取系统总 CPU 核心数

    # 处理 "all" 特殊值
    if affinity_str.strip().lower() == "all":
        return list(range(total_cpus))

    # 拆分逗号分隔的段（如 "0-3,5" → ["0-3", "5"]）
    segments = affinity_str.strip().split(",")
    for seg in segments:
        seg = seg.strip()
        if not seg:  # 跳过空段（如 ",,0-3" 中的空值）
            continue

        try:
            # 场景1：单核心（如 "5"）
            if "-" not in seg:
                cpu = int(seg)
                # 校验核心数是否合法（避免超出系统总核心数）
                if 0 <= cpu < total_cpus:
                    cpu_list.append(cpu)
                continue

            # 场景2：连续范围（如 "0-3"）
            start_str, end_str = seg.split("-", 1)  # 仅拆分第一个 "-"，避免 "0-3-5" 异常
            start = int(start_str.strip())
            end = int(end_str.strip())

            # 校验范围合法性（start <= end，且核心数在合法范围）
            if start > end:
                print(f"Warning: invalid CPU range {seg}")
                continue
            if start < 0 or end >= max_cpu_num:
                print(f"Warning: CPU range {seg} exceed total cups {max_cpu_num})")
                continue

            # 生成连续核心列表
            cpu_list.extend(range(start, end + 1))

        except ValueError:
            # 处理非数字字符（如 "0-a"、"abc"）
            print(f"Warning: CPU affinity string contais invalid character {seg}")
            continue
        except Exception as e:
            print(f"parse CPU seg {seg} fail: {str(e)}")
            continue

    # 去重 + 排序（避免重复核心，如 "0-3,2-5" → 去重后 [0,1,2,3,4,5]）
    cpu_list = sorted(list(set(cpu_list)))
    return cpu_list


def compress_continuous(nums: list) -> str:
    if not nums:
        return ""

    ranges = []
    sorted_nums = sorted(nums)
    start = end = sorted_nums[0]

    for num in sorted_nums[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = num

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def get_thread_pid_by_tid(tid: int) -> int | None:
    """根据线程tid获取所属进程的pid"""
    pid: int | None = None
    for proc in psutil.process_iter(["pid"]):
        try:
            threads = proc.threads()
            for thread in threads:
                if thread.id == tid:
                    pid = proc.pid
                    break
            if pid:
                break
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # 跳过无权限/已退出的进程
            continue
    return pid


def get_thread_name_by_tid(tid: int, pid: int | None = None) -> str:
    """根据线程tid获取线程名称"""
    if pid is None:
        pid = get_thread_pid_by_tid(tid)

    if pid is None:
        print(f"Error: can not get pid for thread tid={tid})")
        return ""

    try:
        with open(f"/proc/{pid}/task/{tid}/comm", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: get thread name fail - {e}")
        return ""


def get_thread_cmdline_by_tid(tid: int, pid: int | None = None) -> str:
    """
    根据线程tid获取线程名称
    """

    if pid is None:
        pid = get_thread_pid_by_tid(tid)

    if pid is None:
        print(f"Error: can not get pid for thread(tid={tid})")
        return ""

    try:
        with open(f"/proc/{pid}/task/{tid}/cmdline", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: get thread name fail - {e}")
        return ""


def get_thread_cpus_by_tid(tid: int, pid: int | None = None) -> list:
    """根据线程tid获取线程绑定的cpu"""
    if pid is None:
        pid = get_thread_pid_by_tid(tid)

    if pid is None:
        print(f"Error: can not get pid for thread tid={tid})")
        return []

    try:
        with open(f"/proc/{pid}/task/{tid}/status", "r") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    cpu_str = line.strip().split(":", 1)[1].strip()
                    break
            return parse_cpu_affinity_string(cpu_str)
    except Exception as e:
        print(f"Error: get thread cpus fail - {e}")
        return []


def get_threads_by_pattern(pid: int, pattern: str) -> list[tuple[int, str]]:
    """根据正则表达式形式的线程名获取所有线程"""
    matched_tids = []
    pattern_compiled = re.compile(pattern)

    try:
        process = psutil.Process(pid)
        for thread in process.threads():
            tid = thread.id
            thread_name = get_thread_name_by_tid(tid=tid, pid=pid)
            if pattern_compiled.search(thread_name):
                matched_tids.append((tid, thread_name))
                continue
            thread_name = get_thread_cmdline_by_tid(tid=tid, pid=pid)
            if pattern_compiled.search(thread_name):
                matched_tids.append((tid, thread_name))

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        print(f"Fail to get process {pid}: {e}")

    return matched_tids


def get_tid_by_thread_name(thread_name: str, pid: int | None, process_name: str | None = None) -> list[int]:
    """根据线程名获取线程tid"""
    if pid is None and process_name is not None:
        pids = get_pid_by_process_name(process_name)
        pid = pids[0][0] if pids else None

    if pid is None:
        print(f"Error: can not get pid for thread[{thread_name}])")
        return []

    matched_tids = []
    threads = get_threads_by_pattern(pid=pid, pattern=thread_name)
    for tid, name in threads:
        matched_tids.append(tid)
    return sorted(matched_tids)


def get_pid_by_process_name(
    process_name: str, exact_match: bool = True, parent_name: str | None = None
) -> list[tuple[int, str]]:
    """根据进程名获取进程pid"""
    matched_pids = []
    pattern_compiled = re.compile(process_name)

    try:
        for proc in psutil.process_iter(["pid", "name", "ppid"]):
            try:
                proc_name = proc.info.get("name")
                proc_pid = proc.info.get("pid")
                proc_ppid = proc.info.get("ppid")

                if proc_name is None or proc_pid is None:
                    continue

                name_match: bool = False
                if exact_match:
                    name_match = proc_name == process_name
                else:
                    name_match = bool(pattern_compiled.search(proc_name))

                if not name_match:
                    continue

                if parent_name is not None:
                    if proc_ppid is None:
                        continue
                    try:
                        parent_proc = psutil.Process(proc_ppid)
                        if parent_proc.name() != parent_name:
                            continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                matched_pids.append((proc_pid, proc_name))

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

    except psutil.AccessDenied:
        print("Error: access denied")
    except Exception as e:
        print(f"Error: get pid fail - {e}")

    return matched_pids


def get_process_cpus_by_pid(pid: int) -> list:
    return psutil.Process(pid).cpu_affinity()


def get_all_user_processes() -> list[tuple[int, str]]:
    """
    获取所有用户态进程，返回（pid，name）列表
    """

    all_processes = []
    try:
        for proc in psutil.process_iter(["pid", "name", "ppid"]):
            try:
                pid = proc.info["pid"]
                name = proc.info["name"]
                ppid = proc.info["ppid"]

                if None in (pid, name, ppid):
                    continue

                # 内核进程过滤
                if pid == 0 or pid == 2 or ppid == 2:
                    continue

                all_processes.append((pid, name))

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except psutil.AccessDenied:
        print("Error: access denied")
    except Exception as e:
        print(f"Error: get pid fail - {e}")

    return all_processes


def safe_listdir(path: str) -> list:
    try:
        file_list = os.listdir(path)
        return file_list
    except FileNotFoundError:
        print(f"Error: file not found → {path}")
        return []
    except NotADirectoryError:
        print(f"Error: not a directory → {path}")
        return []
    except PermissionError:
        print(f"Error: no permission → {path}")
        return []
    except TypeError as e:
        print(f"Error: para type error - {e}")
        return []
    except OSError as e:
        print(f"Error: OS Error → {path}, detail: {e.strerror} (error no: {e.errno})")
        return []
    except Exception as e:
        print(f"Unknown Error → {path}, defail: {str(e)}")
        return []


def read_str_param(root_dir, name) -> tuple[str, bool]:
    path = os.path.join(root_dir, name)
    try:
        with open(path, "r") as file:
            return file.read().strip(), True
    except Exception as e:
        print(f"Error: read {path} failed - {e}")
        return "", False


def read_int_param(root_dir, name) -> int:
    param, ret = read_str_param(root_dir, name)
    return int(param) if ret and param.isdigit() else -1


def read_list_param(root_dir, name) -> tuple[list, bool]:
    path = os.path.join(root_dir, name)
    try:
        with open(path, "r") as file:
            list_param: list[int] = []
            list_str = file.read().strip()
            segments = [s.strip() for s in list_str.split(",") if s.strip()]
            for seg in segments:
                # 匹配 "起始-结束" 格式
                try:
                    if "-" in seg:
                        start_str, end_str = seg.split("-")
                        start = int(start_str.strip())
                        end = int(end_str.strip())
                        # 生成连续的CPU ID（包含起始和结束）
                        list_param.extend(range(start, end + 1))
                    else:
                        val = int(seg.strip())
                        list_param.append(val)
                except Exception as e:
                    print(f"Error: cast item in list failed, {path} - {e}")
                    continue
            return list_param, True
    except Exception as e:
        print(f"Error: read {path} failed - {e}")
        return [], False


def execute_command(cmd: list[str]) -> tuple[str, str, int]:
    with subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, err = p.communicate(timeout=60)
    out_str = out.decode()
    err_str = err.decode()
    if err_str and p.returncode != 0:
        print(f"Command '{cmd}' failed with error - {err_str} ")
    return out_str, err_str, p.returncode


def bind_process_to_cpus(pid: int, cpus: list) -> None:
    cpu_list = ",".join(map(str, cpus))
    out, _, return_code = execute_command(["taskset", "-acp", cpu_list, str(pid)])
    if return_code != 0:
        print(f"Failed to bind {pid} to CPU {cpu_list}, info: {out}")


def bind_thread_to_cpus(tid: int, cpus: list) -> None:
    cpu_list = ",".join(map(str, cpus))
    out, _, return_code = execute_command(["taskset", "-cp", cpu_list, str(tid)])
    if return_code != 0:
        print(f"Failed to bind {tid} to CPU {cpu_list}, info: {out}")


def bind_irq_to_cpus(irq_id: int, cpus: list, irq_name: str = "") -> None:
    cpu_mask = CPUMask().from_list(cpus)
    try:
        with open(f"/proc/irq/{irq_id}/smp_affinity_list", "w") as f:
            f.write(f"{cpu_mask}")
    except Exception as e:
        print(f"[Error] faild to bind irq[{irq_id}]({irq_name}) to {cpus} - {e}")


def bind_npu_sq_send_wq_to_cpus(npu_id: int, cpus: list) -> None:
    cpu_mask = CPUMask().from_list(cpus).to_mask()
    try:
        with open(f"/sys/devices/virtual/workqueue/dev{npu_id}_sq_send_wq/cpumask", "w") as f:
            f.write(f"{cpu_mask}")
    except Exception as e:
        print(f"[Error] faild to bind dev{npu_id}_sq_send_wq to {cpus} (mask={cpu_mask})- {e}")


def migrate_process_pages(pid: int, src_numa: list, tgt_numa: int) -> None:
    _, _, return_code = execute_command(
        [
            "migratepages",
            str(pid),
            ",".join(map(str, src_numa)),
            str(tgt_numa),
        ]
    )
    if return_code != 0:
        print(f"Failed to migrate pages for process [{pid}] " f"from source numa {src_numa} to target numa {tgt_numa}")


def get_npu_work_queue_cpus_by_name(wq_name: str) -> list:
    try:
        with open(f"/sys/devices/virtual/workqueue/{wq_name}/cpumask", "r") as f:
            cpu_mask = f.read().strip()
            cpu = CPUMask().from_mask(cpu_mask)
            return cpu.to_list()
    except Exception as e:
        print(f"Error: get workqueue {wq_name} cpus fail - {e}")
        return []


def get_irq_cpus_by_irq_id(irq_id: int) -> list:
    try:
        with open(f"/proc/irq/{irq_id}/smp_affinity_list", "r") as f:
            cpu_str = f.read().strip()
            return parse_cpu_affinity_string(cpu_str)
    except Exception as e:
        print(f"Error: get irq {irq_id} cpus fail - {e}")
        return []


def get_npu_irq_by_name(irq_name: str, npu_id: int) -> list:
    all_irqs: list[str] = []
    try:
        with open("/proc/interrupts") as f:
            for line in f:
                if irq_name in line:
                    irq = line.split(":")[0].strip()
                    all_irqs.append(irq)
    except Exception as e:
        print(f"get interrupts (name={irq_name}) fail - {e}")
        return []

    # todo: 支持npu type获取
    npu_type = "A3"
    if npu_type == "A3":
        card_id = npu_id // 2
        chip_id = npu_id % 2
        info, _, _ = execute_command(["npu-smi", "info", "-t", "board", "-i", str(card_id), "-c", str(chip_id)])
    else:
        info, _, _ = execute_command(["npu-smi", "info", "-t", "board", "-i", str(npu_id)])

    pci_addr = ""
    for line in info.splitlines():
        if "PCIe Bus Info" in line:
            pci_addr = line.split()[-1].lower()
            break
    if not pci_addr:
        print(f"cannot found pci address of npu [{npu_id}].")
        return []

    try:
        npu_irq_list = sorted(
            os.listdir(f"/sys/bus/pci/devices/{pci_addr}/msi_irqs/"),
            key=lambda x: int(x),
        )
    except FileNotFoundError:
        print(f"cannot found msi_irqs folder under /sys/bus/pci/devices/{pci_addr}.")
        return []

    res_irqs: list[int] = []
    for irq in all_irqs:
        if irq in npu_irq_list:
            res_irqs.append(int(irq))
    return res_irqs


def get_allowed_cpu_list() -> list[int]:
    """
    获取允许的cpu列表：
        1. 如果程序运行在容器内，获取当前容器实际生效的cpuset
        2. 如果程序运行在宿主机，获取宿主机的cpuset

    Returns:
        1. 非空列表：成功获取到的允许的cpu列表
        2. 空列表：读取失败，不做限制
    """

    CGROUP_ROOT = "/sys/fs/cgroup"

    # 获取当前进程所在cgroup相对路径
    cgroup_path = ""
    try:
        with open("/proc/self/cgroup", "r") as f:
            line = f.readline().strip()
            parts = line.split(":", 2)
            if len(parts) >= 3:
                cgroup_path = parts[2]
    except Exception:
        return []

    # 拼接 cpuset 真实路径
    v2_path = f"{CGROUP_ROOT}{cgroup_path}/cpuset.cpus.effective"
    v1_path = f"{CGROUP_ROOT}/cpuset{cgroup_path}/cpuset.effective_cpus"

    for p in [v2_path, v1_path]:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    cpu_str = f.read().strip()
                    return parse_cpu_affinity_string(cpu_str)
            except:
                continue

    return []


def is_cpu_online(cpu_id: int) -> bool:
    """
    检查指定CPU是否在线
    """

    # cpu0 永远在线
    if cpu_id == 0:
        return True

    online_path = f"/sys/devices/system/cpu/cpu{cpu_id}/online"
    try:
        with open(online_path, "r") as f:
            return f.read().strip() == "1"

    except Exception as e:
        print(f"Warning: check cpu{cpu_id} online status failed - {e}")
        return True
