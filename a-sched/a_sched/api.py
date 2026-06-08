from __future__ import annotations
from a_sched.engine import affinity_engine as affinity


def group_create(name: str = "") -> int:
    """
    创建新组并返回组ID

    Returns:
        int: 新创建的组ID，如果创建失败返回-1

    Raises:
        RuntimeError: 当组创建失败时可能抛出异常
    """
    return affinity.task.group_create(name=name)


def group_destroy(group_id: int) -> None:
    """
    销毁指定组

    Args:
        group_id: 要销毁的组ID

    Returns:
        None: 不返回值

    Note:
        如果组不存在，静默返回（不抛异常）
    """
    affinity.task.destory_group(group_id=group_id)


def group_add_thread(
    group_id: int,
    tid: int | None = None,
    thread_name: str | None = None,
    pid: int | None = None,
    process_name: str | None = None,
) -> None:
    """
    将线程添加到指定任务组中。

    Args:
        group_id: 目标任务组的ID，必须是已存在的有效组ID。
        tid: 要添加的线程ID，必须是正在运行线程的有效TID。
        thread_name: 线程的唯一名称，用于标识和监控。

    Raises:
        当添加失败时抛出异常
    """
    affinity.task.group_add_thread(
        group_id=group_id, tid=tid, thread_name=thread_name, pid=pid, process_name=process_name
    )


def group_remove_thread(group_id: int, tid: int | None = None, thread_name: str | None = None) -> None:
    """
    从组中移除线程

    Args:
        group_id: 组ID
        tid: 线程ID
        thread_name: 线程名称

    Raises:
        当移除失败时抛出异常
    """
    affinity.task.group_remove_thread(group_id=group_id, tid=tid, thread_name=thread_name)


def group_add_process(
    group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
) -> None:
    """
    将进程添加到指定任务组中。

    Args:
        group_id: 目标任务组的ID。必须是已存在的有效组ID。
        pid: 要添加的进程ID。必须是正在运行进程的有效PID。
        process_name: 进程的唯一名称。用于标识和监控。
        parent_name: 父进程的名称，用于在多个同名进程中过滤指定父进程的子进程。

    Raises:
        当添加失败时抛出异常
    """
    affinity.task.group_add_process(group_id=group_id, pid=pid, process_name=process_name, parent_name=parent_name)


def group_remove_process(
    group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
) -> None:
    """
    从组中移除进程

    Args:
        group_id: 组ID
        pid: 进程ID
        process_name: 进程名称
        parent_name: 父进程的名称，用于在多个同名进程中过滤指定父进程的子进程。

    Raises:
        当移除失败时抛出异常
    """
    affinity.task.group_remove_process(group_id=group_id, pid=pid, process_name=process_name, parent_name=parent_name)


def thread_set_high_priority(
    tid: int | None = None, thread_name: str | None = None, pid: int | None = None, process_name: str | None = None
) -> None:
    """
    指定线程设置高优先级

    Args:
        tid: 线程ID
        thread_name: 线程名称

    Raises:
        当设置失败时抛出异常
    """
    affinity.task.thread_set_high_priority(tid=tid, thread_name=thread_name, pid=pid, process_name=process_name)


def process_bind_npu(
    npu_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
) -> None:
    """
    进程绑定NPU

    Args:
        npu_id: npu的id
        pid: 进程ID
        process_name: 进程名
        parent_name: 父进程的名称，用于在多个同名进程中过滤指定父进程的子进程。

    Raises:
        失败时抛出异常
    """
    affinity.task.process_bind_npu(npu_id=npu_id, pid=pid, process_name=process_name, parent_name=parent_name)


def run_affinity(dry_run: bool = False) -> None:
    """
    运行亲和调度

    Args:
        dry_run: True表示试运行，仅输出亲和方案，不做亲和方案执行；False表示运行亲和调度全流程
    """
    affinity.run(dry_run=dry_run)


def print_affinity() -> None:
    """
    打印亲和组中进程/线程当前实际的亲和信息
    """
    affinity.print_affinity()


def restore_affinity() -> None:
    """
    恢复亲合组中进程/线程原始的亲和信息
    """
    affinity.restore_affinity()


def set_exclude_cpu(cpu_str: str) -> None:
    """
    设置不参与亲和调度的cpu

    Args:
        cpu_str: 字符串格式的cpu列表，示例：0-3, 10, 20
    """
    affinity.config.set_exclude_cpu(cpu_str=cpu_str)


def enable_cpuset_isolate(enable: bool = True) -> None:
    """
    启用/关闭 cpuset隔离
    """
    affinity.config.enable_cpuset = enable
