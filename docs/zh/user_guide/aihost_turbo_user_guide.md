# AIHost-turbo 使用指南

## 1. 项目概述

AIHost-turbo 是一个用于推理框架进程/线程的自适应亲和调度工具。它可以根据当前的硬件拓扑结构和任务配置，自动为进程和线程分配 CPU 亲和资源，实现高效的亲和调度。

### 核心功能

- **进程亲和调度**：将进程绑定到特定的 CPU 核心
- **线程亲和调度**：将线程绑定到特定的 CPU 核心
- **NPU 绑定**：将进程与 NPU 资源关联
- **亲和恢复**：支持恢复原始亲和配置
- **CPU 排除**：支持指定不参与亲和调度的 CPU 核心
- **cpuset 隔离**：支持 cpuset 资源隔离，进一步提升性能隔离效果

---

## 2. 使用步骤

### 2.1 关键进程梳理

在使用工具前，用户需要熟悉推理框架的关键进程，并梳理进程间的关系，将关键进程的名称添加到亲和域内。

**以 vLLM 框架为例**：

关键进程包括：
- `VLLM::EngineCore`：引擎核心进程，负责模型推理的协调调度
- `VLLM::Worker`：工作进程，负责实际的推理计算

用户需要根据实际使用的框架，识别并记录关键进程的名称。

### 2.2 编写绑核脚本

用户需要自定义自适应亲和调度脚本。请参考示例脚本 [affinity_vllm.py](../../a-sched/examples/affinity_vllm.py)，根据实际情况替换进程名称和调整参数配置。

### 2.3 确定并行策略参数

在使用亲和调度工具前，需要根据推理框架的模型启动参数，确定并行策略参数（TP、DP、EP），在后续绑核过程中分别使用 `-tp-size`、`-dp-size`、`-ep` 参数指定。

#### 并行策略说明

| 策略类型 | 说明 | 适用场景 |
|---------|------|---------|
| **TP (Tensor Parallel)** | 张量并行，将模型层内的张量切分到多个设备 | 大模型单卡显存不足、需要提升单卡推理性能 |
| **DP (Data Parallel)** | 数据并行，在不同设备上处理不同的数据批次 | 提升吞吐量、多卡训练场景 |
| **EP (Expert Parallel)** | 专家并行，将 MoE 模型的专家分配到不同设备 | MoE（Mixture of Experts）模型架构 |

**参数确定方法**：

1. 从推理框架启动命令中获取并行参数
2. 根据硬件资源配置计算：
   - 单机场景：`TP × DP ≤ NPU 总数`
   - 跨机场景：`TP × DP × 机器数 ≤ 总 NPU 数`

### 2.4 运行绑核脚本

执行绑核脚本，应用亲和调度策略：

```bash
python examples/affinity_vllm.py -tp-size 8 -dp-size 1 -dp-size-local 1 -dp-start-rank 0 -ep -r
```

#### 参数说明

| 参数 | 说明 |
|------|------|
| `-tp-size` | 张量并行大小 |
| `-dp-size` | 数据并行大小 |
| `-dp-size-local` | 当前节点的数据并行大小 |
| `-dp-start-rank` | 当前节点的起始全局 rank（跨机场景） |
| `-ep` | 启用专家并行（仅 vLLM） |
| `-r` | 运行亲和调度 |
| `-d` | 试运行亲和调度（仅做方案决策，不做方案执行） |
| `-p` | 打印当前亲和信息 |
| `-restore` | 恢复原始亲和配置 |

---

## 3. 注意事项

1. **权限要求**：修改进程/线程亲和性需要 root 权限
2. **进程状态**：添加进程或线程时，需要确保目标进程/线程正在运行
3. **跨机部署**：跨机部署时，需要正确设置 `dp-start-rank` 参数
4. **配置备份**：执行绑核后会自动备份亲和配置，可用 `-restore` 参数恢复
5. **调试建议**：建议先使用 `-d` 参数验证配置正确性，再执行实际绑定

---

## 4. FAQ

### 如何确认绑核是否生效？

可以通过以下命令查看进程的 CPU 亲和性设置：

```bash
# 查看进程亲和性
taskset -p <pid>
```

### 如何查看当前亲和配置？

使用 `-p` 参数打印当前亲和信息（仅查看不执行）：

```bash
python examples/affinity_vllm.py -tp-size 8 -dp-size 1 -dp-size-local 1 -dp-start-rank 0 -ep -p
```

### 绑核失败怎么办？

请检查以下内容：

1. 是否具有 root 权限
2. 目标进程/线程是否正在运行
3. CPU 资源是否充足
4. 硬件拓扑配置是否正确
5. 进程名称是否正确匹配

### 如何取消绑核配置？

使用 `-restore` 参数恢复原始配置：

```bash
python examples/affinity_vllm.py -tp-size 8 -dp-size 1 -dp-size-local 1 -dp-start-rank 0 -ep -restore
```

### 如何排除特定 CPU 核心？

使用 `set_exclude_cpu` API 指定不参与调度的 CPU 核心：

```python
import a_sched
a_sched.set_exclude_cpu("0-3,10,20")
```

### 如何启用 cpuset 隔离？

```python
import a_sched
a_sched.enable_cpuset_isolate(enable=True)
```

---