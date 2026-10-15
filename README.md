# MECO: An Efficient MCU–Edge–Cloud System for Factory Maintenance

## 场景

MECO 面向工厂中的传送带监测、泵故障诊断和机械臂故障诊断，将设备端 MCU、边缘 CPU 与云端 GPU 组织为分层维护系统。每个请求包含设备身份、当前传感器观测和设备维护知识。维护规则与操作说明相对稳定，温度、电流、压力、振动等观测随任务变化。

MCU 负责采集状态与本地控制，边缘小模型完成上游诊断，部分任务继续交由 GPU 上的强模型处理。系统通过迁移已经计算的 KV 状态减少下游重复 prefill，并在诊断返回后检查设备身份、会话与动作条件，使模型推理与设备控制形成完整闭环。

![Figure 1：传送带、泵和机械臂的工厂维护场景](assets/figure1-scene.png)

*Figure 1. 工厂中的三类维护任务：传送带监测、泵故障诊断和机械臂故障诊断。*

## 系统框架

![Figure 2：MECO 的 MCU、边缘与云端系统框架](assets/figure2-framework.png)

*Figure 2. MECO 系统总览。蓝色 A–E 为请求服务路径，黑色 1–3 为调度控制路径。*

设备观测经请求网关进入边缘服务。CPU 小模型先处理维护上下文与当前任务，能够在边缘完成的结果直接送至完成校验模块；需要进一步诊断的任务将完整 KV 状态交给映射模块，转换为 GPU 强模型的层与注意力头布局，再由强模型继续生成。两条路径均通过响应格式、设备会话和动作条件检查，最终由 MCU 返回执行回执。本地控制循环在推理期间保持独立运行。

Figure 2 展示论文的整体设计，包括上下文复用与调度控制。本代码包实现其中的完整 KV 级联服务路径：CPU、GPU 各自采用 FIFO 队列，两个阶段可处理不同请求并行推进；每个进入 GPU 的任务迁移完整上游 KV，并从未消费的 token 与后续提示继续推理。论文图中的跨请求前缀存储与局部性感知重排不属于此精简包。当前分流依据输入中的 `sensor_valid`：无效读数由 CPU 完成，其余进入 GPU。

## 实现的硬件和软件条件

硬件平台、操作系统及上下游模型与论文第 3.6 节的部署配置对应如下。

| 层级 | 硬件与资源 | 软件与职责 |
|---|---|---|
| MCU | 3 个 STM32F407 仿真节点；每节点 1 MiB Flash、128 KiB SRAM | Renode 运行 C 固件；通过 USART2 socket 传输观测、命令与 CRC 回执 |
| Edge CPU | Intel Core i9-14900KF，分配 4 核 | Python 异步网关；Qwen2.5-Coder-1.5B-Instruct 上游推理 |
| Cloud GPU | NVIDIA GeForce RTX 4090，24 GB 显存 | Qwen2.5-Coder-7B-Instruct 下游推理；接收并映射上游 KV |
| 操作系统 | Ubuntu 24.04.4 LTS | 承载网关、模型服务与设备仿真 |

本代码包使用 Python 3.12，CPU 以 FP32 推理，GPU 以 FP16 推理，KV 通过 safetensors 以 FP16 序列化。网关、CPU 与 GPU 分别使用独立 Python 环境，依赖版本固定在 `requirements/` 中。MCU 固件使用 ARM GNU 工具链编译，仿真环境为 Renode 1.17.0。部署入口支持 Linux/WSL2，模型服务通过本机 HTTP 通信。默认 CPU 亲和性为上游推理 0–3、GPU 服务主机线程 4–7、Renode 8–9，可在配置中调整。

## 代码库结构介绍

```text
MECO-system-core-20261015/
├── meco/
│   ├── strategy/       # 输入分流、阶段 FIFO、token 续接边界
│   ├── inference/      # CPU/GPU 推理、模型身份、HTTP 服务
│   ├── kv/             # 跨模型 KV 映射与 RoPE 变换
│   ├── transport/      # 元数据与 KV 字节的封装、解包、摘要
│   ├── device/         # CRC 协议、异步 UART、响应格式检查
│   ├── runtime/        # 任务流水线、服务进程管理、系统启动
│   └── __main__.py     # python -m meco 入口
├── mcu/
│   ├── firmware/       # STM32F407 固件与链接脚本
│   ├── renode/         # 仿真平台描述
│   └── build.sh        # 按设备 ID 编译固件
├── config/             # 解释器、权重目录、端口与 CPU 绑定配置
├── requirements/       # 网关、CPU、GPU 的依赖文件
├── assets/             # README 使用的论文 Figure 1、2、6
└── README.md
```

`meco/runtime/pipeline.py` 串联观测上传、CPU 推理、可选 GPU 续接与 MCU 回执；`meco/strategy/continuation.py` 确定已缓存前缀和待处理后缀，`meco/kv/mapper.py` 执行跨模型映射；`meco/device/uart.py` 将请求和返回动作绑定到具体设备会话。

## 结果展示

以下展示论文报告的结果。合成负载包含三类设备、12 份维护规则文档，每类设备有 4 种配置；每个种子生成 256 个任务，到达过程为每秒 0.6 个请求的泊松过程，共评估 5 个种子。延迟从计划到达时刻计至 MCU 回执，包含正确和错误结果；先计算各个种子的分位数，再对 5 个种子取算术平均。

![Figure 6：FIFO 与 MECO 的全部回执端到端延迟对比](assets/figure6-latency.png)

*Figure 6. 全部回执的端到端延迟，单位为秒。*

相较 FIFO，MECO 的 P50、P90、P95 和 P99 延迟分别降低 **29.28%、24.72%、20.37% 和 21.82%**。整体准确率由 FIFO 的 **90.23%** 变为 MECO 的 **89.84%**，下降 **0.39 个百分点**。FIFO 对进入 GPU 的完整上下文重新 prefill，MECO 迁移上游 KV 后续接；这些结果展示整体延迟与准确率的取舍，不单独归因于调度收益。

**Table 1. 论文报告的组件平均功率与峰值功率。**

| 组件 | 平均功率（W） | 峰值功率（W） |
|---|---:|---:|
| MCU（3 个节点合计） | 0.06435 | 0.10890 |
| Edge CPU | 61.60 | 66.96 |
| Cloud GPU（RTX 4090） | 78.64 | 256.56 |

MCU 数值为三个仿真节点的场景估计，CPU 与 GPU 数值来自各自传感器窗口的汇总。报告的平均功率比约为 **1:957:1222**，表示组件量级差异，不代表同步测得的系统总功率。

## 简单的部署说明

在 Ubuntu 或 WSL2 中准备 CUDA 驱动、Renode 和 `arm-none-eabi-gcc`，将代码解压到无空格的路径，并在代码根目录创建三个 Python 环境：

```bash
python3.12 -m venv "$HOME/meco-env-gateway"
python3.12 -m venv "$HOME/meco-env-cpu"
python3.12 -m venv "$HOME/meco-env-gpu"

"$HOME/meco-env-gateway/bin/pip" install -r requirements/gateway.txt
"$HOME/meco-env-cpu/bin/pip" install -r requirements/cpu.txt
"$HOME/meco-env-gpu/bin/pip" install -r requirements/gpu.txt
```

预先准备模型与新映射权重。模型放在 `$HOME/.cache/huggingface/hub` 的标准 snapshot 目录中，使用以下固定版本；映射目录需包含当前新权重方案的 `manifest.json` 和 28 个 layer safetensors 文件。模型身份和映射权重版本由 `meco/inference/models.py` 固定，服务启动时进行校验。

| 模型 | 固定版本（revision） |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |

复制示例配置，填写 CPU/GPU 解释器、Renode 路径和适用的 CPU 绑定，指定映射权重目录后启动：

```bash
cp config/system.example.json config/system.json
export MECO_MAPPER_DIR=/path/to/newweights/mapper

"$HOME/meco-env-gateway/bin/python" -m meco \
  --config config/system.json \
  --tasks /path/to/tasks.json \
  --output /path/to/new-runtime-output
```

`tasks.json` 接受任务列表或包含 `requests` 列表的对象。每项提供 `uid`、`device_id`、`sample_sequence`、`family`、`document_id`、`observation` 和 `input_ids`；设备 1/2/3 分别对应传送带、泵和机械臂。`input_ids` 由上述固定 tokenizer 对任务提示编码，`arrival_ms` 可指定相对到达时间，生成预算为 96 tokens。输出目录须尚不存在。启动入口会编译三份固件、启动模型服务与 Renode，并将逐任务响应和 MCU 回执写入 `responses.jsonl`。
