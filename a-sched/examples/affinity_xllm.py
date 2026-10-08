import argparse
from dataclasses import dataclass

import a_sched as affinity
import a_sched.utils as utils


@dataclass
class Config:
    dp_size_local: int = 0
    local_nodes: int = 0
    npu_start: int = 0

    @classmethod
    def parse_from_args(cls, args: argparse.Namespace):
        return cls(
            dp_size_local=args.dp_size_local,
            local_nodes=args.local_nodes,
            npu_start=args.npu_start,
        )


def get_xllm_process(config: Config) -> list[tuple[int, str]]:
    xllm_processes = sorted(utils.get_pid_by_process_name(process_name="xllm", top_level=True))
    process_count = len(xllm_processes)

    assert process_count == config.local_nodes, (
        f"size of xllm main processes not equal to local_nodes, local_nodes={config.local_nodes}, "
        f"size of xllm main processes: {process_count}"
    )
    return xllm_processes


def add_affinity_tasks(config: Config) -> None:
    for index, (pid, name) in enumerate(get_xllm_process(config=config)):
        group_id = affinity.group_create()
        affinity.group_add_process(group_id=group_id, pid=pid, process_name=name)
        affinity.process_bind_npu(npu_id=config.npu_start + index, pid=pid)


def cmd_run(config: Config, dry_run: bool = False, drop_caches_first: bool = False) -> None:
    try:
        add_affinity_tasks(config)
        affinity.set_schedule_strategy(isolate="cluster")
        affinity.set_drop_caches(drop_caches_first and not dry_run)
        affinity.run_affinity(dry_run=dry_run)
    except Exception as error:
        print(f"run affinity schedule failed, {str(error)}")


def cmd_print(config: Config) -> None:
    try:
        add_affinity_tasks(config)
        affinity.print_affinity()
    except Exception as error:
        print(f"print current affinity info failed, {str(error)}")


def cmd_restore() -> None:
    affinity.restore_affinity()


def main(args: argparse.Namespace) -> None:
    config = Config.parse_from_args(args)
    if args.run:
        cmd_run(config, drop_caches_first=args.drop_caches)
    elif args.dry_run:
        cmd_run(config, dry_run=True)
    elif args.print:
        cmd_print(config)
    elif args.restore:
        cmd_restore()
    else:
        print("Not found command to execute!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="xllm推理框架自适应亲和隔离调度工具",
        epilog="使用示例: python affinity_xllm.py --dp-size-local 2 --local-nodes 8 -r",
    )
    parser.add_argument("-r", "--run", action="store_true", help="运行亲和调度")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="试运行亲和调度（仅做方案决策，不做方案执行）",
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="打印亲和组中进程和线程当前实际的亲和信息",
    )
    parser.add_argument("--restore", action="store_true", help="恢复进程和线程亲和性到调度前状态")
    parser.add_argument("--dp-size-local", type=int, required=True, help="本机data parallel大小")
    parser.add_argument("--local-nodes", type=int, required=True, help="本机xllm节点数量")
    parser.add_argument("--npu-start", type=int, default=0, help="起始NPU ID, 默认0")
    parser.add_argument(
        "--drop-caches",
        action="store_true",
        help="在执行亲和调度前清理页缓存",
    )
    main(parser.parse_args())
