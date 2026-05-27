import argparse
from dataclasses import dataclass
import a_sched as affinity


@dataclass
class Config:
    dp_size_local: int = 0
    dp_start_rank: int = 0
    dp_size: int = 0
    tp_size: int = 0
    enable_ep: bool = False

    @classmethod
    def parse_from_args(cls, args: argparse.Namespace):
        return cls(
            dp_size_local=args.dp_size_local if args.dp_size_local > 0 else args.dp_size,
            dp_start_rank=args.dp_start_rank,
            dp_size=args.dp_size,
            tp_size=args.tp_size,
            enable_ep=args.enable_ep
        )


def get_engine_core_process_name(dp_global: int, config: Config) -> str:
    process_name = "VLLM::EngineCore"
    if config.dp_size > 1:
        process_name += f"_DP{dp_global}"
    return process_name


def get_worker_process_name(dp_global: int, tp: int, config: Config) -> str:
    process_name = "VLLM::Worker"
    if config.dp_size > 1 and config.enable_ep:
        process_name += f"_DP{dp_global}"
    if config.tp_size > 1:
        process_name += f"_TP{tp}"
    if config.enable_ep:
        ep = dp_global * config.tp_size + tp
        process_name += f"_EP{ep}"
    return process_name


def add_affinity_tasks(config: Config) -> None:
    for dp_local in range(config.dp_size_local):
        dp_global = dp_local + config.dp_start_rank

        # 每个DP域创建1个亲和组
        group_id = affinity.group_create()

        # 添加 engine core 进程到亲和组
        engine_core_name = get_engine_core_process_name(dp_global=dp_global, config=config)
        affinity.group_add_process(group_id=group_id, process_name=engine_core_name)

        for tp in range(config.tp_size):
            # 添加 worker 进程到亲和组
            worker_name = get_worker_process_name(dp_global=dp_global, tp=tp, config=config)
            affinity.group_add_process(group_id=group_id, process_name=worker_name, parent_name=engine_core_name)

            # worker 进程绑定对应的 npu
            npu_id = dp_local * config.tp_size + tp
            affinity.process_bind_npu(npu_id=npu_id, process_name=worker_name, parent_name=engine_core_name)


def cmd_run(config: Config, dry_run: bool = False):
    try:
        add_affinity_tasks(config)
        affinity.run_affinity(dry_run=dry_run)

    except Exception as e:
        print(f"run affinity schedule failed, {str(e)}")


def cmd_print(config: Config):
    try:
        add_affinity_tasks(config)
        affinity.print_affinity()

    except Exception as e:
        print(f"print current affinity info failed, {str(e)}")


def cmd_restore():
    affinity.restore_affinity()


def main(args: argparse.Namespace):
    config = Config.parse_from_args(args)
    if args.run:
        cmd_run(config)
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
        prog="自适应亲和隔离调度工具",
        description="用于vllm推理框架自适应亲和隔离调度",
        epilog="使用示例: python affinity_vllm.py -tp-size 2 -dp-size 4 -r",
    )
    parser.add_argument(
        "-r", "--run", action="store_true",
        help="运行亲和调度"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="试运行亲和调度，仅做方案决策，不做方案执行"
    )
    parser.add_argument(
        "-p", "--print", action="store_true",
        help="打印亲和组进程/线程当前实际的亲和信息"
    )
    parser.add_argument(
        "-restore", "--restore", action="store_true",
        help="恢复进程和线程亲和性到调度前状态"
    )
    parser.add_argument(
        "-tp-size", "--tp-size", type=int, required=True,
        help="张量并行(tensor parallel)大小"
    )
    parser.add_argument(
        "-dp-size", "--dp-size", type=int, required=True,
        help="数据并行(data parallel)大小"
    )
    parser.add_argument(
        "-dp-size-local", "--dp-size-local", type=int, default=0,
        help="当前节点数据并行(data parallel)大小"
    )
    parser.add_argument(
        "-dp-start-rank", "--dp-start-rank", type=int, default=0,
        help="当前节点数据并行(data parallel)起始rank"
    )
    parser.add_argument(
        "-ep", "--enable_ep", action="store_true",
        help="使能专家并行(expert parallel)"
    )
    args = parser.parse_args()
    main(args)
