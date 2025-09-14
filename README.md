# RISE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

**R**eason-**I**nspire-**S**trengthen-**E**xpertise for Large Language Models.  

📑 Our Paper: [RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning](https://arxiv.org/abs/2508.13229)

---

## 🇨🇳 中文版

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
    pip install -r requirements.txt
    ```

4.  **安装 LLaMA-Factory (仅用于可选的 SFT 阶段)**:
    ```bash
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

### 2. 可选预热 SFT (使用 LLaMA-Factory)

此阶段是可选的，使用 LLaMA-Factory 对基座模型进行监督微调。  

### 3. RISE-CoT 训练 (自定义训练框架)

使用我们自定义的训练框架进行思维链训练，注入推理启发能力。  

### 4. RISE-R1 训练 (自定义训练框架)

使用我们自定义的强化学习框架进行能力巩固和专业化训练。  

### 5. 推理与验证

使用推理脚本加载最终模型，验证其性能。  

## 🤝 贡献

欢迎任何形式的贡献！  

## 📄 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。  

## 🙏 致谢

感谢开源社区的巨大贡献。  

---

## 🇬🇧 English Version

This project aims to enhance the reasoning ability of large language models in complex tasks through innovative training strategies. It adopts the HuggingFace dataset format, supports multimodal inputs (text and images), and provides a complete training pipeline, including optional model warm-up and the core RISE training stages.

> **Code is Coming Soon!**  
> The project is under active development. Please click "Watch" and "Star" to get the latest updates!

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Features](#-features)
- [Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
  - [1. Dataset Format](#1-dataset-format)
  - [2. Optional Warm-up SFT (with LLaMA-Factory)](#2-optional-warm-up-sft-with-llama-factory)
  - [3. RISE-CoT Training (Custom Framework)](#3-rise-cot-training-custom-framework)
  - [4. RISE-R1 Training (Custom Framework)](#4-rise-r1-training-custom-framework)
  - [5. Inference & Evaluation](#5-inference--evaluation)
- [Contribution](#-contribution)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🏗 Architecture Overview

The workflow is divided into the following stages, with model warm-up being optional:

1. **Data Preparation** (HuggingFace dataset format)  
2. **Optional Model Warm-up** (**LLaMA-Factory**) → `sft_model` (optional)  
3. **Chain-of-Thought Training** (**RISE-CoT Framework**) → `rise_cot_model`  
4. **Reinforcement Training** (**RISE-R1 Framework**) → `rise_r1_model`  
5. **Inference & Evaluation** (**Inference Scripts**)  

## ✨ Features

- **Multimodal Support**: Accepts text and image inputs for complex reasoning tasks.  
- **Flexible Preprocessing**: HuggingFace dataset format with fields `problem`, `image`, `image_path`, `answer`, and `target`.  
- **Innovative RISE Strategy**: Two-stage training framework:  
  - **RISE-CoT (Reason-Inspire)**: Enhances reasoning and inspiration abilities.  
  - **RISE-R1 (Strengthen-Expertise)**: Consolidates and specializes skills via reinforcement learning.  
- **Optional Warm-up**: SFT stage is provided but can be skipped.  

## ⚙️ Installation

Requirements:
- Python 3.10+
- PyTorch (CUDA)
- HuggingFace Transformers & Datasets
- Git

```bash
git clone https://github.com/HSH55/RISE.git
cd RISE
conda create -n rise python=3.10
conda activate rise
pip install -r requirements.txt
````

(Optional, for SFT stage):

```bash
git submodule update --init --recursive
cd LLaMA-Factory
pip install -e .
cd ..
```

## 🚀 Quick Start

### 1. Dataset Format

We use the HuggingFace dataset format with fields: `problem`, `image`, `image_path`, `answer`, `target`.

### 2. Optional Warm-up SFT (with LLaMA-Factory)

Supervised fine-tuning of the base model (optional).

### 3. RISE-CoT Training (Custom Framework)

Chain-of-thought training to inject reasoning and inspiration abilities.

### 4. RISE-R1 Training (Custom Framework)

Reinforcement training to consolidate and professionalize the model’s skills.

### 5. Inference & Evaluation

Load the final model with our inference scripts and validate its performance.

## 🤝 Contribution

We welcome all forms of contribution!

## 📄 License

This project is licensed under MIT. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

We sincerely thank the open-source community for their contributions.

