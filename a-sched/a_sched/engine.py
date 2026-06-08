from a_sched.affinity_domain import AffinityDomainManager
from a_sched.cpuset import CpusetManager
from a_sched.task import TaskManager
from a_sched.backup import AffinityBackup
from a_sched.config import AffinityConfig
from a_sched.scheduler import Scheduler
from a_sched.strategy.hierarchical_balance import HierarchicalBalanceScheduler
import a_sched.utils as utils


class AffinityEngine:
    """亲和调度引擎"""

    def __init__(self) -> None:
        self._scheduler: Scheduler | None = None
        self._init_engine()

    def _init_engine(self) -> None:
        self.config = AffinityConfig()
        self.task = TaskManager()
        self.domain = AffinityDomainManager(config=self.config)
        self.cpuset = CpusetManager(config=self.config, task=self.task, domain=self.domain)
        self.backup = AffinityBackup(task=self.task, domain=self.domain)

    def run(self, dry_run: bool = False) -> None:
        """
        运行亲和调度

        Args:
            dry_run: True表示试运行，仅输出亲和方案，不做亲和方案执行；False表示运行亲和调度全流程
        """

        print("\n-------------------------- Affinity Schedule Begin --------------------------")

        try:
            print("\nStarting build affinity domain...")
            self.domain.build_affinity_domain()

            print("The affinity domain is as follows:")
            self._print_affinity_domain()

            print("\nThe affinity info before schedule is as follows:")
            self.print_affinity()

            print("\nStarting plan affinity...")
            self._plan()

            print("The affinity plan is as follows:")
            self._print_affinity_plan()

            if not dry_run:
                print("\nStarting execute affinity...")
                self._execute()

                print("\nThe affinity info after schedule is as follows:")
                self.print_affinity()

            print("\nAffinity schedule SUCCESS!")

        except Exception as e:
            print(f"\nAffinity schedule FAILED! Error: {e}.")

        print("---------------------------- Affinity Schedule End ----------------------------")

    def _plan(self) -> None:
        # 刷新高优先级线程和npu互相绑定关系
        self.task.update_high_prio_thread_bind_npu()

        # 扫描背景任务
        self.task.scan_background_tasks()

        # 使用分层均衡亲和调度策略决策亲和方案
        self._scheduler = HierarchicalBalanceScheduler(self.config, self.domain, self.task)
        if not self._scheduler.schedule():
            raise RuntimeError("Plan affinity failed!")

    def _execute(self) -> None:
        # 执行前先备份当前亲和信息
        self.backup_affinity()
        # 停止CPU硬件中断自动均衡
        self._stop_irq_balance()
        # 根据亲和方案绑定cpu
        self._bind_cpus()
        # 迁移进程内存到新numa节点
        self._bind_memory()

    def _print_affinity_domain(self) -> None:
        print("------------------------------- Affinity Domain -------------------------------")
        self.domain.print_all()

    def _print_affinity_plan(self) -> None:
        print("-------------------------------- Affinity Plan --------------------------------")
        self.task.print_all()

    def _stop_irq_balance(self) -> None:
        print("\nStopping irqbalance service...")
        _, _, return_code = utils.execute_command(["systemctl", "is-active", "--quiet", "irqbalance"])
        if return_code == 0:
            utils.execute_command(["systemctl", "stop", "irqbalance"])
            print("the irqbalance service has been stopped.")

    def _bind_cpus(self) -> None:
        print("\nStarting bind cpus...")

        # 绑定普通亲和组任务
        for group in self.task.groups.values():
            for task in group.get_normal_prio_tasks():
                task.bind_cpu()

        # 动态隔离高优先级任务绑定的cpu
        self._isolate_cpus()

        # 绑定高优先级亲和组任务
        for group in self.task.groups.values():
            for task in group.get_high_prio_tasks():
                task.bind_cpu()

        # 绑定背景任务
        self._bind_background_tasks()

    def _bind_background_tasks(self) -> None:
        normal_cpus = self.task.background_tasks_cpus
        if not normal_cpus:
            print("[BackgrondTask] No target CPUs found, skipping background tasks binding.")
            return

        failed = []
        bound = 0

        for pid, name in self.task.background_processes.items():
            try:
                utils.bind_process_to_cpus(pid, normal_cpus)
                bound += 1
            except Exception as e:
                failed.append((pid, name, str(e)))

        if failed:
            print(f"[BackgroundTask] Failed to bind {len(failed)} processes.")
            for pid, name, err in failed:
                print(f"  - process[{pid}]({name}): {err}")

        total = len(self.task.background_processes)
        print(
            f"[BackgroundTask] Bind {bound}/{total} processes to CPUs {utils.compress_continuous(normal_cpus)}"
            f" (failed: {len(failed)})"
        )

    def _isolate_cpus(self) -> None:
        isolate_cpus = []
        shared_cpus = []
        for group in self.task.groups.values():
            isolate_cpus.extend(group.isolate_cpus.to_list())
            shared_cpus.extend(group.cpus.to_list())
        self.cpuset.isolate_cpus(isolate_cpus, shared_cpus)

    def _bind_memory(self) -> None:
        print("\nStarting bind memory...")
        for _, group in self.task.groups.items():
            for _, process in group.process_tasks.items():
                if not process.numa:
                    print(f"can not get target numa of process [{process.task_id}]")
                    continue
                tgt_numa = process.numa[0]
                src_numa = self.domain.get_all_numas_id()
                if not src_numa:
                    print(f"can not get source numa of process [{process.task_id}]")
                    continue
                print(
                    f"migrating pages of process [{process.task_id}] from source numa {src_numa} to target numa [{tgt_numa}]"
                )
                try:
                    utils.migrate_process_pages(pid=process.task_id, src_numa=src_numa, tgt_numa=tgt_numa)
                except Exception as e:
                    print(f"Failed to migrate pages for process [{process.task_id}]: {str(e)}")

    def print_affinity(self) -> None:
        """打印亲和任务中进程/线程当前实际的亲和信息"""

        print("--------------------------- Current Affinity Status ---------------------------")
        self.backup.print_affinity()

    def backup_affinity(self) -> None:
        """备份亲和任务中进程/线程当前亲和信息"""

        print("\nStarting backup current affinity...")
        self.backup.backup_affinity()

    def restore_affinity(self) -> None:
        """恢复亲和任务中进程/线程原始亲和信息"""

        print("\n-------------------------- Affinity Resotre Begin ---------------------------")
        print("Starting restore affinity...")

        self.backup.restore_affinity()

        print("\nThe affinity after restore is as follows:")
        self.print_affinity()

        print(f"\nAffinity restore SUCCESS!")
        print("---------------------------- Affinity Resotre End -----------------------------")

    def reset(self) -> None:
        self._init_engine()


affinity_engine = AffinityEngine()
