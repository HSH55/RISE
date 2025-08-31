# RISE-LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

**R**einforced **I**terative **S**trategy **E**nhancement for Large Language Models.

本项目旨在通过结合 CoT（思维链）和 R1（策略增强）训练，来提升模型在复杂推理和策略规划任务上的性能。项目使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行高效的**预热 SFT（监督微调）**，而核心的 **RISE-CoT** 和 **RISE-R1** 训练阶段则由我们自定义的高效训练框架实现。

> **Code is Coming Soon!**
> 本项目正在积极开发中，代码即将发布。请点击右上角的 "Watch" 和 "Star" 以获取最新更新！

## 📋 目录

- [架构概述](#-架构概述)
- [功能特性](#-功能特性)
- [安装依赖](#-安装依赖)
- [🚀 快速开始](#-快速开始)
  - [1. 数据集制作](#1-数据集制作)
  - [2. 预热 SFT (使用 LLaMA-Factory)](#2-预热-sft-使用-llama-factory)
  - [3. RISE-CoT 训练 (自定义训练框架)](#3-rise-cot-训练-自定义训练框架)
  - [4. RISE-R1 训练 (自定义训练框架)](#4-rise-r1-训练-自定义训练框架)
  - [5. 推理与验证](#5-推理与验证)
- [贡献](#-贡献)
- [许可证](#-许可证)
- [致谢](#-致谢)

## 🏗 架构概述

本项目的工作流清晰地分为以下几个阶段，其中仅预热 SFT 阶段依赖于 LLaMA-Factory：

1.  **数据准备** (`dataset.json`)
2.  **模型预热** (**LLaMA-Factory**) -> `sft_model`
3.  **思维链训练** (**RISE-CoT Framework**) -> `rise_cot_model`
4.  **策略增强训练** (**RISE-R1 Framework**) -> `rise_r1_model`
5.  **推理验证** (**Inference Scripts**)

## ✨ 功能特性

- **混合训练框架**: 结合了成熟的 LLaMA-Factory SFT 流程与我们自主研发的高效训练框架。
- **创新的 RISE 策略**: 实现了核心的两阶段训练方法：
  - **RISE-CoT**: 我们的自定义训练框架，专注于培养模型的思维链和分步推理能力。
  - **RISE-R1**: 我们的自定义训练框架，基于强化学习信号，进一步优化模型的策略和输出质量。
- **高性能与灵活性**: 自定义训练框架为您的特定需求优化，提供更好的控制和效率。

## ⚙️ 安装依赖

在开始之前，请确保您的环境满足以下要求：
- Python 3.9+
- PyTorch (CUDA)
- Git

1.  **克隆本仓库**:
    ```bash
    git clone https://github.com/your-username/RISE-LLM.git
    cd RISE-LLM
    ```

2.  **安装项目依赖**:
    ```bash
    # 安装核心依赖、自定义训练框架所需的库等
    pip install -r requirements.txt
    ```

3.  **安装 LLaMA-Factory (仅用于 SFT 阶段)**:
    ```bash
    # 可选：如果选择将 LLaMA-Factory 作为子模块嵌入
    git submodule update --init --recursive
    cd LLaMA-Factory
    pip install -e .
    cd ..
    ```
    *注：具体的依赖安装细节将在代码库就绪后完善。*

## 🚀 快速开始

### 1. 数据集制作

我们使用特定格式的 JSON 文件进行训练。您需要准备一个包含 `"instruction"`、`"input"`（可选）和 `"output"` 字段的数据集。

**示例数据格式 (`dataset.json`)**:
```json
[
  {
    "instruction": "请解释一下牛顿第一定律。",
    "input": "",
    "output": "牛顿第一定律，也称为惯性定律..."
  }
]
```
*详细的数据集构建脚本和指南将在代码中提供。*

### 2. 预热 SFT (使用 LLaMA-Factory)

此阶段使用 LLaMA-Factory 对基座模型进行高效的监督微调，为后续训练打下基础。

```bash
# 进入 LLaMA-Factory 目录或使用其命令行工具
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/base_model \
    --dataset your_sft_dataset \
    --template llama2 \
    --finetuning_type lora \
    --output_dir ./output/sft_model \
    --per_device_train_batch_size 4 \
    ... # 其他 SFT 参数
```
*此阶段产生 `sft_model`，作为下一步的输入。*

### 3. RISE-CoT 训练 (自定义训练框架)

**这是本项目核心的创新训练阶段之一。** 使用我们自定义的训练框架，基于 SFT 模型继续训练，注入思维链能力。

```bash
# 使用项目自定义的训练脚本
python rise_cot/train.py \
    --model_name_or_path ./output/sft_model \
    --data_path ./data/cot_dataset.json \
    --output_dir ./output/rise_cot_model \
    --batch_size 8 \
    --learning_rate 2e-5
    ... # 自定义框架的参数
```
*此阶段产生 `rise_cot_model`，具备强大的推理能力。*

### 4. RISE-R1 训练 (自定义训练框架)

**这是本项目另一个核心的创新训练阶段。** 使用我们自定义的强化学习框架，对 CoT 模型进行策略增强。本阶段的实现参考了 [VisualRFT](https://github.com/fuliucansheng/VisualRFT) 的工作，采用了类似的拒绝采样与策略优化流程，并针对我们的任务进行了适配与改进。

```bash
# 使用项目自定义的 RL 训练脚本
python rise_r1/train.py \
    --model_name_or_path ./output/rise_cot_model \
    --reward_model /path/to/reward_model \
    --data_path ./data/rl_dataset.json \
    --output_dir ./output/rise_r1_model \
    ... # 自定义 RL 框架的参数
```
*此阶段产生最终的 `rise_r1_model`。*

### 5. 推理与验证

使用我们提供的推理脚本加载最终模型，验证其性能。

```bash
# 使用项目自定义的推理脚本
python scripts/inference.py \
    --model_path ./output/rise_r1_model \
    --prompt "你的问题 here"
```

## 🤝 贡献

我们热烈欢迎任何形式的贡献！包括但不限于：
- 📖 完善文档
- 🐛 提交 Bug 报告
- 💡 提出新功能建议
- 🔧 提交 Pull Request

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) (待创建) 了解详情。

## 📄 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目建立在众多巨人肩膀之上，我们由衷感谢开源社区带来的巨大贡献。

- **核心框架**: 感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 为我们提供了极其高效便捷的 SFT 微调基础，极大地加速了项目初期开发。
- **方法参考**: 我们的 RISE-R1 (策略增强) 训练阶段深受 [VisualRFT](https://github.com/fuliucansheng/VisualRFT) 工作的启发，借鉴了其基于拒绝采样的强化学习微调思想，在此表示诚挚的感谢。
- **社区与灵感**: 感谢 Hugging Face、Meta (LLaMA) 等为社区提供了强大的基座模型和生态系统。同时也感谢所有为本项目提供建议和帮助的贡献者。
