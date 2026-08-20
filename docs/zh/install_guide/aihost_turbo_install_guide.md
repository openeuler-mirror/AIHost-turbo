# AIHost-turbo 安装指南

## 一、环境依赖

### Python 依赖

```bash
pip install psutil
```

### 系统依赖

```bash
dnf install numactl
```

## 二、部署方式

### 方式 1：源码安装

源码方式通过设置 `PYTHONPATH` 直接引用本地代码

```bash
# 1. 克隆源码
git clone https://gitcode.com/openeuler/AIHost-turbo.git

# 2. 进入 a-sched 目录并配置环境变量
cd AIHost-turbo/a-sched
export PYTHONPATH=$PWD

# 3. 进入示例脚本目录即可使用
cd examples
```

完成后即可直接运行 `affinity_vllm.py`。