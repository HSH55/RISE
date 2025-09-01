# RISE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

**R**eason-**I**nspire-**S**trengthen-**E**xpertise for Large Language Models.

本项目旨在通过创新的训练方法提升大语言模型在复杂推理任务上的能力。项目采用 HuggingFace 数据集格式，支持多模态输入（文本和图像），并提供完整的训练流程，包括可选的模型预热和核心的 RISE 训练阶段。

> **Code is Coming Soon!**
> 本项目正在积极开发中，代码即将发布。请点击右上角的 "Watch" 和 "Star" 以获取最新更新！

## 📋 目录

- [架构概述](#-架构概述)
- [功能特性](#-功能特性)
- [安装依赖](#-安装依赖)
- [🚀 快速开始](#-快速开始)
  - [1. 数据集格式](#1-数据集格式)
  - [2. 可选预热 SFT (使用 LLaMA-Factory)](#2-可选预热-sft-使用-llama-factory)
  - [3. RISE-CoT 训练 (自定义训练框架)](#3-rise-cot-训练-自定义训练框架)
  - [4. RISE-R1 训练 (自定义训练框架)](#4-rise-r1-训练-自定义训练框架)
  - [5. 推理与验证](#5-推理与验证)
- [贡献](#-贡献)
- [许可证](#-许可证)
- [致谢](#-致谢)

## 🏗 架构概述

本项目的工作流清晰地分为以下几个阶段，其中模型预热阶段是可选的：

1.  **数据准备** (HuggingFace 格式数据集)
2.  **可选模型预热** (**LLaMA-Factory**) -> `sft_model` (可选)
3.  **思维链训练** (**RISE-CoT Framework**) -> `rise_cot_model`
4.  **策略增强训练** (**RISE-R1 Framework**) -> `rise_r1_model`
5.  **推理验证** (**Inference Scripts**)

## ✨ 功能特性

- **多模态支持**: 支持文本和图像输入，适用于复杂的多模态推理任务。
- **灵活的预处理**: 使用 HuggingFace 数据集格式，包含 problem、image、image_path、answer 和 target 字段。
- **创新的 RISE 策略**: 实现了核心的两阶段训练方法：
  - **RISE-CoT (Reason-Inspire)**: 我们的自定义训练框架，专注于培养模型的思维推理和启发能力。
  - **RISE-R1 (Strengthen-Expertise)**: 我们的自定义训练框架，通过强化学习进一步巩固和专业化模型的技能。
- **可选预热**: 提供可选的模型预热阶段，可根据需要跳过直接进行 RISE 训练。

## ⚙️ 安装依赖

在开始之前，请确保您的环境满足以下要求：
- Python 3.10+
- PyTorch (CUDA)
- HuggingFace Transformers & Datasets
- Git

1.  **克隆本仓库**:
    ```bash
    git clone https://github.com/HSH55/RISE.git
    cd RISE
    ```

2.  **创建并激活 Python 3.10+ 环境** (推荐使用 conda):
    ```bash
    conda create -n rise python=3.10
    conda activate rise
    ```

3.  **安装项目依赖**:
    ```bash
    # 安装核心依赖、自定义训练框架所需的库等
    pip install -r requirements.txt
    ```

4.  **安装 LLaMA-Factory (仅用于可选的 SFT 阶段)**:
    ```bash
    # 可选：如果选择将 LLaMA-Factory 作为子模块嵌入
    git submodule update --init --recursive
    cd LLaMA-Factory
    pip install -e .
    cd ..
    ```
    *注：具体的依赖安装细节将在代码库就绪后完善。*

## 🚀 快速开始

### 1. 数据集格式

我们使用 HuggingFace 数据集格式，需要包含以下五个字段：

- **problem**: 问题描述文本
- **image**: 图像数据（可以是 PIL 图像或图像张量）
- **image_path**: 图像文件路径
- **answer**: 模型生成的答案
- **target**: 目标答案或参考答案

**示例数据集结构**:
```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "problem": ["问题描述1", "问题描述2", ...],
    "image": [image1, image2, ...],  # PIL 图像或张量
    "image_path": ["path/to/image1.jpg", "path/to/image2.png", ...],
    "answer": ["模型答案1", "模型答案2", ...],
    "target": ["目标答案1", "目标答案2", ...]
})
```
*详细的数据集构建指南将在代码中提供。*

### 2. 可选预热 SFT (使用 LLaMA-Factory)

此阶段是可选的，使用 LLaMA-Factory 对基座模型进行监督微调，为后续训练打下基础。如果已有合适的预训练模型，可以跳过此步骤。

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
*此阶段产生 `sft_model`，作为下一步的输入（可选）。*

### 3. RISE-CoT 训练 (自定义训练框架)

**Reason-Inspire 阶段**: 使用我们自定义的训练框架，基于基础模型或 SFT 模型进行思维链训练，注入推理启发能力。

```bash
# 使用项目自定义的训练脚本
# 可以从基础模型或 SFT 模型开始
python rise_cot/train.py \
    --model_name_or_path /path/to/base_or_sft_model \
    --data_path /path/to/huggingface/dataset \
    --output_dir ./output/rise_cot_model \
    --batch_size 8 \
    --learning_rate 2e-5
    ... # 自定义框架的参数
```
*此阶段产生 `rise_cot_model`，具备强大的推理和启发能力。*

### 4. RISE-R1 训练 (自定义训练框架)

**Strengthen-Expertise 阶段**: 使用我们自定义的强化学习框架，对 CoT 模型进行能力巩固和专业化训练。本阶段的实现参考了 [VisualRFT](https://github.com/fuliucansheng/VisualRFT) 的工作，采用了类似的拒绝采样与策略优化流程，并针对我们的任务进行了适配与改进。

```bash
# 使用项目自定义的 RL 训练脚本
python rise_r1/train.py \
    --model_name_or_path ./output/rise_cot_model \
    --reward_model /path/to/reward_model \
    --data_path /path/to/huggingface/dataset \
    --output_dir ./output/rise_r1_model \
    ... # 自定义 RL 框架的参数
```
*此阶段产生最终的 `rise_r1_model`，具备专业级的推理能力。*

### 5. 推理与验证

使用我们提供的推理脚本加载最终模型，验证其性能。支持多模态输入（文本和图像）。

```bash
# 使用项目自定义的推理脚本
python scripts/inference.py \
    --model_path ./output/rise_r1_model \
    --problem "你的问题描述" \
    --image_path "path/to/image.jpg"  # 可选图像输入
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

- **核心框架**: 感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 为我们提供了高效便捷的 SFT 微调基础（可选阶段）。
- **方法参考**: 我们的 RISE-R1 (Strengthen-Expertise) 训练阶段深受 [VisualRFT](https://github.com/fuliucansheng/VisualRFT) 工作的启发，借鉴了其基于拒绝采样的强化学习微调思想，在此表示诚挚的感谢。
- **社区与灵感**: 感谢 Hugging Face 提供了强大的数据集和模型生态系统。同时也感谢所有为本项目提供建议和帮助的贡献者。
