
# RISE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

**R**eason-**I**nspire-**S**trengthen-**E**xpertise for Large Language Models.

---

## 🌍 English (default)

This project aims to enhance large language models' capabilities on complex reasoning tasks through innovative training strategies. It adopts the HuggingFace dataset format, supports multimodal inputs (text and images), and provides a complete training pipeline, including an optional model warm-up stage and the core RISE training phases.

> **Code is Coming Soon!**  
> This project is under active development, and the code will be released soon. Please click "Watch" and "Star" in the top-right corner to stay updated!

### 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)  
- [Features](#-features)  
- [Installation](#-installation)  
- [🚀 Quick Start](#-quick-start)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Acknowledgments](#-acknowledgments)  

### 🏗 Architecture Overview

1. **Data Preparation** (HuggingFace dataset format)  
2. **Optional Model Warm-up** (**LLaMA-Factory**) → `sft_model` (optional)  
3. **Chain-of-Thought Training** (**RISE-CoT Framework**) → `rise_cot_model`  
4. **Reinforcement Training** (**RISE-R1 Framework**) → `rise_r1_model`  
5. **Inference & Evaluation** (**Inference Scripts**)  

### ✨ Features

- Multimodal support (text + images)  
- Flexible preprocessing with HuggingFace datasets  
- Two-stage RISE training strategy:
  - **RISE-CoT (Reason-Inspire)** → chain-of-thought reasoning  
  - **RISE-R1 (Strengthen-Expertise)** → RL-based reinforcement  
- Optional warm-up with LLaMA-Factory  

### ⚙️ Installation

```bash
git clone https://github.com/HSH55/RISE.git
cd RISE
conda create -n rise python=3.10
conda activate rise
pip install -r requirements.txt
````

(Optional) Install LLaMA-Factory:

```bash
git submodule update --init --recursive
cd LLaMA-Factory
pip install -e .
cd ..
```

### 🚀 Quick Start

* Dataset fields: `problem`, `image`, `image_path`, `answer`, `target`
* Optional SFT warm-up with LLaMA-Factory
* Run **RISE-CoT training**
* Run **RISE-R1 training**
* Perform inference with multimodal input

### 🤝 Contributing

We welcome:

* Documentation improvements
* Bug reports
* Feature requests
* Pull requests

### 📄 License

MIT License (see [LICENSE](LICENSE)).

### 🙏 Acknowledgments

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* [VisualRFT](https://github.com/fuliucansheng/VisualRFT)
* Hugging Face ecosystem

---

<details>
<summary>🇨🇳 中文说明 (点击展开)</summary>

## 简介

本项目旨在通过创新的训练方法提升大语言模型在复杂推理任务上的能力。项目采用 HuggingFace 数据集格式，支持多模态输入（文本和图像），并提供完整的训练流程，包括可选的模型预热和核心的 RISE 训练阶段。

> **代码即将发布！**
> 项目正在积极开发中，请点击右上角的 "Watch" 和 "Star" 以获取最新更新！

### 📋 目录

* [架构概述](#-架构概述)
* [功能特性](#-功能特性)
* [安装依赖](#-安装依赖)
* [🚀 快速开始](#-快速开始)
* [贡献](#-贡献)
* [许可证](#-许可证)
* [致谢](#-致谢)

### 🏗 架构概述

1. **数据准备** (HuggingFace 格式数据集)
2. **可选模型预热** (**LLaMA-Factory**) → `sft_model` (可选)
3. **思维链训练** (**RISE-CoT 框架**) → `rise_cot_model`
4. **策略增强训练** (**RISE-R1 框架**) → `rise_r1_model`
5. **推理验证** (**推理脚本**)

### ✨ 功能特性

* 支持文本和图像的多模态输入
* HuggingFace 数据集格式，字段包括：`problem`, `image`, `image_path`, `answer`, `target`
* 两阶段创新训练方法：

  * **RISE-CoT (Reason-Inspire)**: 思维链推理
  * **RISE-R1 (Strengthen-Expertise)**: 基于强化学习的能力巩固
* 可选的预热阶段（LLaMA-Factory）

### ⚙️ 安装依赖

```bash
git clone https://github.com/HSH55/RISE.git
cd RISE
conda create -n rise python=3.10
conda activate rise
pip install -r requirements.txt
```

（可选）安装 LLaMA-Factory：

```bash
git submodule update --init --recursive
cd LLaMA-Factory
pip install -e .
cd ..
```

### 🚀 快速开始

* 数据集字段：`problem`, `image`, `image_path`, `answer`, `target`
* 可选的 SFT 预热（LLaMA-Factory）
* 运行 **RISE-CoT 训练**
* 运行 **RISE-R1 训练**
* 使用推理脚本验证模型性能（支持多模态输入）

### 🤝 贡献

欢迎：

* 完善文档
* 提交 Bug 报告
* 提出新功能建议
* 提交 Pull Request

### 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

### 🙏 致谢

* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* [VisualRFT](https://github.com/fuliucansheng/VisualRFT)
* Hugging Face 社区与生态

</details>

