

# [RISE: Enhancing VLM Image Annotation with Self-Supervised Reasoning](https://arxiv.org/abs/2508.13229)

# RISE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

**R**eason-**I**nspire-**S**trengthen-**E**xpertise for Large Language Models.

This project aims to enhance large language models’ capabilities on complex reasoning tasks through innovative training methods. It adopts the HuggingFace dataset format, supports multimodal inputs (text and images), and provides a complete training pipeline, including optional model warm-up and the core RISE training stages.

> **Code is Coming Soon!**
> The project is under active development, and code will be released soon. Please click "Watch" and "Star" in the upper right corner to stay updated!

## 📋 Table of Contents

* [Architecture Overview](#-architecture-overview)
* [Features](#-features)
* [Installation](#-installation)
* [🚀 Quick Start](#-quick-start)

  * [1. Dataset Format](#1-dataset-format)
  * [2. Optional Warm-up SFT (using LLaMA-Factory)](#2-optional-warm-up-sft-using-llama-factory)
  * [3. RISE-CoT Training (Custom Training Framework)](#3-rise-cot-training-custom-training-framework)
  * [4. RISE-R1 Training (Custom Training Framework)](#4-rise-r1-training-custom-training-framework)
  * [5. Inference & Evaluation](#5-inference--evaluation)
* [Contributing](#-contributing)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

## 🏗 Architecture Overview

The workflow of this project is divided into the following stages, with model warm-up being optional:

1. **Data Preparation** (HuggingFace dataset format)
2. **Optional Model Warm-up** (**LLaMA-Factory**) -> `sft_model` (optional)
3. **Chain-of-Thought Training** (**RISE-CoT Framework**) -> `rise_cot_model`
4. **Reinforcement Training** (**RISE-R1 Framework**) -> `rise_r1_model`
5. **Inference & Evaluation** (**Inference Scripts**)

## ✨ Features

* **Multimodal Support**: Accepts both text and image inputs, suitable for complex multimodal reasoning tasks.
* **Flexible Preprocessing**: Uses HuggingFace dataset format with fields: `problem`, `image`, `image_path`, `answer`, and `target`.
* **Innovative RISE Strategy**: Implements a two-stage training approach:

  * **RISE-CoT (Reason-Inspire)**: A custom training framework focusing on reasoning and inspiration capabilities.
  * **RISE-R1 (Strengthen-Expertise)**: A custom reinforcement learning framework to consolidate and specialize model expertise.
* **Optional Warm-up**: Provides optional supervised fine-tuning before RISE training.

## ⚙️ Installation

Ensure your environment meets the following requirements:

* Python 3.10+
* PyTorch (CUDA)
* HuggingFace Transformers & Datasets
* Git

1. **Clone the repository**:

   ```bash
   git clone https://github.com/HSH55/RISE.git
   cd RISE
   ```

2. **Create and activate Python 3.10+ environment** (conda recommended):

   ```bash
   conda create -n rise python=3.10
   conda activate rise
   ```

3. **Install dependencies**:

   ```bash
   # Install core dependencies and libraries for the custom training framework
   pip install -r requirements.txt
   ```

4. **Install LLaMA-Factory (for optional SFT stage only)**:

   ```bash
   # Optional: add LLaMA-Factory as a submodule
   git submodule update --init --recursive
   cd LLaMA-Factory
   pip install -e .
   cd ..
   ```

   *Note: Detailed dependency installation instructions will be provided once the codebase is ready.*

## 🚀 Quick Start

### 1. Dataset Format

We use the HuggingFace dataset format, which must include the following five fields:

* **problem**: problem description text
* **image**: image data (PIL image or tensor)
* **image\_path**: image file path
* **answer**: model-generated answer
* **target**: reference/ground-truth answer

**Example Dataset Structure**:

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "problem": ["problem description 1", "problem description 2", ...],
    "image": [image1, image2, ...],  # PIL image or tensor
    "image_path": ["path/to/image1.jpg", "path/to/image2.png", ...],
    "answer": ["model answer 1", "model answer 2", ...],
    "target": ["target answer 1", "target answer 2", ...]
})
```

*A detailed dataset construction guide will be provided in the code.*

### 2. Optional Warm-up SFT (using LLaMA-Factory)

This optional stage performs supervised fine-tuning on the base model with LLaMA-Factory. If you already have a suitable pre-trained model, you may skip this step.

```bash
# Run within LLaMA-Factory directory or use its CLI
python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /path/to/base_model \
    --dataset your_sft_dataset \
    --template llama2 \
    --finetuning_type lora \
    --output_dir ./output/sft_model \
    --per_device_train_batch_size 4 \
    ... # other SFT parameters
```

*This stage produces `sft_model`, which can be used as input for the next step (optional).*

### 3. RISE-CoT Training (Custom Training Framework)

**Reason-Inspire Stage**: Uses our custom training framework to perform chain-of-thought training on the base or SFT model, injecting reasoning and inspiration capabilities.

```bash
# Using the custom training script
# Start from either base model or SFT model
bash RISE-COT/2B_COCO.sh
```

*This stage produces `rise_cot_model`, with strong reasoning and inspiration ability.*

### 4. RISE-R1 Training (Custom Training Framework)

**Strengthen-Expertise Stage**: Uses our custom reinforcement learning framework to refine the CoT model. This stage is inspired by [VisualRFT](https://github.com/fuliucansheng/VisualRFT), adopting a similar rejection sampling and policy optimization process, adapted for our tasks.

```bash
# Using the custom RL training script
bash RISE-R1/2B_COCO.sh
```

*This stage produces the final `rise_r1_model`, with expert-level reasoning ability.*

### 5. Inference & Evaluation

Use our inference script to load the final model and evaluate its performance with multimodal inputs (text and images).

```bash
# Using the custom inference script
python scripts/inference.py \
    --model_path ./output/rise_r1_model \
    --problem "your problem description" \
    --image_path "path/to/image.jpg"  # optional image input
```

## 🤝 Contributing

We warmly welcome contributions of any kind, including:

* 📖 Documentation improvements
* 🐛 Bug reports
* 💡 New feature suggestions
* 🔧 Pull Requests

Please check [CONTRIBUTING.md](CONTRIBUTING.md) (to be created) for details.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project stands on the shoulders of giants, and we sincerely thank the open-source community for their contributions.

* **Core Framework**: Thanks to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for providing an efficient SFT foundation (optional stage).
* **Method Reference**: Our RISE-R1 (Strengthen-Expertise) training stage is heavily inspired by [VisualRFT](https://github.com/fuliucansheng/VisualRFT), which introduced rejection sampling-based RL fine-tuning.
* **Community & Inspiration**: Thanks to Hugging Face for their powerful dataset and model ecosystem, and to all contributors who offered advice and support.

