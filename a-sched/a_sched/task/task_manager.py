from __future__ import annotations

from a_sched.task.task_base import Task, PriorityLevel
from a_sched.task.thread_task import ThreadTask
from a_sched.task.process_task import ProcessTask
from a_sched.task.npu_task import create_npu_task
from a_sched.task.task_group import TaskGroup
from a_sched.affinity_domain import AffinityDomainManager
import a_sched.utils as utils


class TaskManager:
    """调度任务管理"""

    def __init__(self, domain: AffinityDomainManager) -> None:
        self.groups: dict[int, TaskGroup] = {}  # group_id -> task group
        self.process_to_npu: dict[int, int] = {}  # pid -> npu_id
        self.background_processes: dict[int, str] = {}  # 背景进程
        self.background_tasks_cpus: list[int] = []  # 背景任务分配的CPU
        self._current_group_index: int = 0
        self._domain = domain

    def group_create(self, name: str = "") -> int:
        group_id = self._current_group_index
        self.groups[group_id] = TaskGroup(group_id=group_id, name=name)
        self._current_group_index += 1
        return group_id

    def destory_group(self, group_id: int) -> None:
        self.groups.pop(group_id)

    def group_add_thread(
        self,
        group_id: int,
        tid: int | None = None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("add thread failed, either tid or thread_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"add thread failed, group (id={group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"add thread failed, process (pid={pid}, name={process_name}) not found")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"add thread failed, thread (tid={tid}, name={thread_name}) not found")

        group.thread_tasks[tid] = ThreadTask(group_id=group_id, tid=tid, pid=pid, name=thread_name)

    def group_remove_thread(
        self,
        group_id: int,
        tid: int | None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("remove thread failed, either tid or thread_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"remove thread failed, group({group_id}) not found")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"remove thread failed, thread (tid={tid}, name={thread_name}) not found")

        group.thread_tasks.pop(tid, None)

    def group_add_process(
        self, group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("add process failed, either pid or process_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"add process failed, group (id={group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"add process failed, process (pid={pid}, name={process_name}) not found")

        group.process_tasks[pid] = ProcessTask(group_id=group_id, pid=pid, name=process_name)

    def group_remove_process(
        self, group_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("remove process failed, either pid or process_name must be set")

        group = self.groups.get(group_id, None)
        if group is None:
            raise ValueError(f"remove process failed, group({group_id}) not found")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"remove process failed, process (pid={pid}, name={process_name}) not found")

        group.process_tasks.pop(pid)

    def thread_set_high_priority(
        self,
        tid: int | None = None,
        thread_name: str | None = None,
        pid: int | None = None,
        process_name: str | None = None,
    ) -> None:
        if tid is None and thread_name is None:
            raise ValueError("set high priority failed, either tid or thread_name must be set")

        if tid is None and thread_name is not None:
            tids = utils.get_tid_by_thread_name(thread_name=thread_name, pid=pid, process_name=process_name)
            tid = tids[0] if tids else None

        if tid is None:
            raise ValueError(f"set high priority failed, thread (tid={tid}, name={thread_name}) not found")

        for _, group in self.groups.items():
            task = group.thread_tasks.get(tid)
            if task is not None:
                task.priority = PriorityLevel.HIGH
                break

    def process_bind_npu(
        self, npu_id: int, pid: int | None = None, process_name: str | None = None, parent_name: str | None = None
    ) -> None:
        if pid is None and process_name is None:
            raise ValueError("process bind npu failed, either pid or process_name must be set")

        if pid is None and process_name is not None:
            pids = utils.get_pid_by_process_name(process_name=process_name, parent_name=parent_name)
            pid = pids[0][0] if pids else None

        if pid is None:
            raise ValueError(f"process bind npu failed, process (pid={pid}, name={process_name}) not found")

        group = self.find_group_by_pid(pid)
        if group is None:
            raise ValueError(f"process bind npu failed, not found group of process (pid={pid}, name={process_name})")

        self.process_to_npu[pid] = npu_id
        npu_task = create_npu_task(group_id=group.group_id, npu_id=npu_id)
        npu_task.bind_pid = pid
        group.npu_tasks[npu_id] = npu_task

    def get_group(self, group_id: int) -> TaskGroup | None:
        return self.groups.get(group_id, None)

    def find_group_by_pid(self, find_pid: int) -> TaskGroup | None:
        for _, group in self.groups.items():
            for pid in group.process_tasks:
                if pid == find_pid:
                    return group
        return None

    def get_high_prio_tasks_of_group(self, group_id: int) -> list[Task]:
        group = self.get_group(group_id)
        return group.get_high_prio_tasks() if group is not None else []

    def print_all(self) -> None:
        for _, group in self.groups.items():
            self.print_task_group(group)

    def print_task_group(self, group: TaskGroup) -> None:
        print(f"{group}")
        for _, process in group.process_tasks.items():
            print(f"  - {process}")
        for _, thread in group.thread_tasks.items():
            print(f"  - {thread}")
        for _, npu in group.npu_tasks.items():
            npu.print()

    def update_high_prio_thread_bind_npu(self) -> None:
        for _, group in self.groups.items():
            for _, thread in group.thread_tasks.items():
                if thread.priority != PriorityLevel.HIGH:
                    continue
                thread.bind_npu = self.process_to_npu.get(thread.pid)
                if thread.bind_npu is None:
                    continue
                npu_task = group.npu_tasks.get(thread.bind_npu)
                if npu_task is not None:
                    npu_task.bind_high_prio_thread.append(thread)

    def scan_background_tasks(self) -> None:
        """
        扫描背景任务
        """

        user_pids = set()
        for _, group in self.groups.items():
            for pid in group.process_tasks.keys():
                user_pids.add(pid)
            for _, npu_task in group.npu_tasks.items():
                user_pids.update(npu_task.processes)

        background_tasks = utils.get_all_user_processes()
        for pid, name in background_tasks:
            if pid not in user_pids:
                self.background_processes[pid] = name

        print(f"[BackgroundTask] Scanned {len(self.background_processes)} processes")


    def init_npu(self) -> None:
        for _, group in self.groups.items():
            for _, npu_task in group.npu_tasks.items():
                npu_task.init_npu()

    def print_actual_affinity(self) -> None:
        """打印亲和任务中进程/线程当前实际的亲和信息"""

        for group in self.groups.values():
            print(f"TaskGroup[{group.group_id}]: {f'name={group.name}' if group.name else ''}")
            for task in group.all_tasks:
                task.print_actual_affinity(self._domain)
