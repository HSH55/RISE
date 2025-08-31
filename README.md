# RISE-LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

**R**einforced **I**terative **S**trategy **E**nhancement for Large Language Models.

本项目旨在通过结合 CoT（思维链）和 R1（策略增强）训练，来提升模型在复杂推理和策略规划任务上的性能。整个流程基于强大的训练框架 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 实现。

> **Code is Coming Soon!**
> 本项目正在积极开发中，代码即将发布。请点击右上角的 "Watch" 和 "Star" 以获取最新更新！

## 📋 目录

- [功能特性](#-功能特性)
- [安装依赖](#-安装依赖)
- [🚀 快速开始](#-快速开始)
  - [1. 数据集制作](#1-数据集制作)
  - [2. RISE-CoT 训练](#2-rise-cot-训练)
  - [3. RISE-R1 训练](#3-rise-r1-训练)
  - [4. 推理与验证](#4-推理与验证)
- [贡献](#-贡献)
- [许可证](#-许可证)
- [致谢](#-致谢)

## ✨ 功能特性

- **统一的训练流程**: 基于 LLaMA-Factory，提供了从 SFT 到强化学习的高效、可配置训练 pipeline。
- **RISE 训练策略**: 实现了创新的两阶段训练方法：
  - **RISE-CoT**: 专注于培养模型的思维链和分步推理能力。
  - **RISE-R1**: 基于增强学习信号，进一步优化模型的策略和输出质量。
- **易于使用**: 提供清晰的脚本和配置示例，帮助用户快速复现和应用。

## ⚙️ 安装依赖

在开始之前，请确保您的环境满足以下要求：
- Python 3.9 或更高版本
- PyTorch (CUDA 版本，与您的 GPU 驱动匹配)
- Git

1. **克隆本仓库**:
   ```bash
   git clone https://github.com/your-username/RISE-LLM.git
   cd RISE-LLM
