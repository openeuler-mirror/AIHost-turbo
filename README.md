# AIHost-turbo

## 简介

**AIHost-turbo** 是一款针对 AI 训练/推理场景 HostBound 问题的性能调优工具。通过感知 Host 侧资源使用情况，对关键线程进行绑核优化，有效排除资源干扰，显著提升系统性能。

致力于为 AI 推理和训练场景构建极致性能加速解决方案。

### 核心组件

| 组件名称 | 功能描述 |
| --- | --- |
| **a-sched** | 亲和性调度器，提供线程绑核优化能力，通过优化CPU调度，解决host侧性能瓶颈 |

## ⚙️ 功能介绍

### a-sched 功能

a-sched 提供线程绑核优化能力，当前已支持的功能如下：

| 功能名称 | 功能描述 |
| --- | --- |
| **线程绑核** | 支持对关键线程进行 CPU 亲和性绑定，避免资源竞争 |
| **任务分组** | 提供任务组管理能力，支持按组进行资源调度和优化 |
| **动态调度** | 根据实时资源使用情况动态调整线程调度策略 |
| **CPU排除** | 支持指定不参与亲和调度的 CPU 核心 |
| **cpuset 隔离** | 支持 cpuset 资源隔离，进一步提升性能隔离效果 |

## 📦 安装指南

介绍 AIHost-turbo 的环境依赖、软件包获取和安装方法，请参见《[AIHost-turbo 安装指南](./docs/zh/install_guide/aihost_turbo_install_guide.md)》。

## 📘 使用指南

AIHost-turbo 的使用方法，请参见《[AIHost-turbo 使用指南](./docs/zh/user_guide/aihost_turbo_user_guide.md)》。

## ❓ FAQ

**Q: AIHost-turbo 支持哪些操作系统？**

A: 当前支持 Linux 操作系统。

**Q: 如何确认绑核是否生效？**

A: 可以通过 `taskset -p <pid>` 命令查看进程的 CPU 亲和性设置。

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/openeuler/AIHost-turbo/issues)，我们会尽快回复。感谢您的支持。

## 🙏 致谢

本工具由华为公司的下列部门联合贡献：
🔹 昇腾计算基础软件开发部
🔹 操作系统部

感谢来自社区的每一个 PR，欢迎贡献！