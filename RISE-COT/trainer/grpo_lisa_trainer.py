# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import io
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
from PIL import Image
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
from transformers import TextIteratorStreamer
from threading import Thread
import re

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
def newprompt_build(text):
    # 匹配 bounding box - 增加容错
    bbox_pattern = r'bounding box \[(.*?)\]'
    bbox_match = re.search(bbox_pattern, text)
    if not bbox_match:
        return text  # 匹配失败返回原文本
    bounding_box = f"[{bbox_match.group(1)}]"

    # 匹配 contains 后的内容 - 增加容错
    contains_pattern = r'contains (.*?)(?:You|$)'
    contains_match = re.search(contains_pattern, text)
    if not contains_match:
        return text  # 匹配失败返回原文本
    contains_content = contains_match.group(1).strip()

    # 安全提取对象名称
    object_name = contains_content.split(', which')[0].split('.')[0]

    new_prompt = f"""Describe why the region is identified as the target object. The region's coordinates are {bounding_box}, and it contains {contains_content} Analyze its shape, color, texture, and contextual clues. Using these visual and contextual features, explain why this region is recognized as {object_name}. Provide a clear Chain-of-Thought (CoT) to justify the conclusion."""
    return new_prompt

def remove_floats(text):
    # 匹配坐标格式的正则表达式，包括可能的空格和换行符
    coordinate_pattern = r'\[?\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]?'
    # 移除匹配到的坐标
    cleaned_text = re.sub(coordinate_pattern, '', text)
    # 匹配浮点数的正则表达式
    float_pattern = r'[-+]?\d*\.\d+([eE][-+]?\d+)?'
    # 使用 re.sub 替换掉所有匹配的浮点数
    cleaned_text = re.sub(float_pattern, '', cleaned_text)

    return cleaned_text


def process_text(text):
    # 去除多余的换行符
    processed_text = re.sub(r'\n\s*\n', '\n\n', text)
    # 去除段落开头的多余空格
    processed_text = re.sub(r'^\s+', '', processed_text, flags=re.MULTILINE)
    # 去除段落结尾的多余空格
    processed_text = re.sub(r'\s+$', '', processed_text, flags=re.MULTILINE)
    # 去除多余的空行
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
    return processed_text


def save_results(completions, cleaned_responses, step_number, save_dir):
    """
    将 completions 和 cleaned_responses 写入同一个文件，并保存前缀为当前步数的结果。

    Args:
        completions (list): 模型生成的原始回复列表。
        cleaned_responses (list): 后处理后的干净回复列表。
        step_number (int): 当前步骤编号。
        save_dir (str): 结果保存目录。
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 构造文件名
    file_name = f"step_{step_number}_results.txt"
    file_path = os.path.join(save_dir, file_name)

    # 写入结果到文件
    with open(file_path, 'w', encoding='utf-8') as file:
        for i, (completion, cleaned_response) in enumerate(zip(completions, cleaned_responses)):
            file.write(f"Sample {i + 1}:\n")
            file.write(f"Completion: {completion}\n")
            file.write(f"Cleaned Response: {cleaned_response}\n")
            file.write("=" * 50 + "\n")

    print(f"Results for step {step_number} have been saved to {file_path}")


class Qwen2VLGRPOTrainer_lisa(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 12845056,
            min_pixels: Optional[int] = 3136,
            step: Optional[int] = 0,
            attn_implementation: str = "flash_attention_2",
            best_iou: Optional[float] = 0.0
    ):
        # 新增初始化代码
        self.step = step
        self.best_iou = best_iou

        # # 设置回调的 trainer 属性
        # for callback in self.callback_handler.callbacks:
        #     callback.set_trainer(self)
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        # 在加载模型时显式指定 torch_dtype
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                # 修改模型加载部分为：
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs
                )
                # from peft import PeftModel
                # model = PeftModel.from_pretrained(
                #     model,
                #     "/data2/HSH/LLM-Dojo-main/model/TRANSTATION_nothink_4shot_100",
                # )
                # print("LoRA加载成功")

            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            print('lora用了')
            model = get_peft_model(model, peft_config)
        # 确保模型被移动到 GPU
        model.to('cuda')

        # Reference model
        if peft_config is None:
            if is_deepspeed_zero3_enabled():
                if "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Aria" in model_id:
                    self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            else:
                # If PEFT configuration is not provided, create a reference model based on the initial model.
                print("deepspeed没有被正确使用")
                self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # 确保参考模型也被移动到 GPU
        if self.ref_model is not None:
            print('参考模型也被移动到 GPU')
            self.ref_model.to('cuda')

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, use_fast=True)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,

        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                       image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _generate_responses(self, inputs):
        """
        生成模型的回复
        """
        # 确保模型和处理器已经加载
        if self.step <= -10:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/data2/HSH/models--Qwen--Qwen2.5-VL-72B-Instruct/model",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                # device_map="auto",
            )
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            processor = AutoProcessor.from_pretrained("/data2/HSH/models--Qwen--Qwen2.5-VL-72B-Instruct/model",
                                                      min_pixels=min_pixels, max_pixels=max_pixels)

        else:
            model = self.model
            processor = self.processing_class
        device = model.device

        # 修改逻辑

        # 提取输入数据
        prompts = [x["prompt"] for x in inputs]
        # prompts_text = [maybe_apply_chat_template(example, processor)["prompt"].replace("this distribution.","this distribution concisely in under 100 words.") for example in inputs]
        prompts_text = [maybe_apply_chat_template(example, processor)["prompt"] for example in inputs]

        # print(prompts_text)

        images = [x["image"] for x in inputs]
        prompt_inputs = processor(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # 确保 prompt_inputs 中的所有张量都在 GPU 上
        for key in prompt_inputs.keys():
            if isinstance(prompt_inputs[key], torch.Tensor):
                prompt_inputs[key] = prompt_inputs[key].to(device)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        generation_config_1 = copy.deepcopy(self.generation_config)
        # generation_config_1.num_return_sequences = 8
        # generation_config_1.temperature = 0.7
        # generation_config_1.repetition_penalty = 1.2

        # print(generation_config_1)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                generation_config=generation_config_1
            )

        # Decode the generated completions
        completions = processor.batch_decode(prompt_completion_ids, skip_special_tokens=True)

        # 后处理
        cleaned_responses = [
            remove_floats(process_text(self.extract_assistant_content(text)))
            for text in completions
        ]
        print(f"模型输出是：{cleaned_responses}")

        return cleaned_responses


    def extract_assistant_content(self, text):
        match = re.search(r'assistant\n(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    def _enhance_prompts(self, inputs):
        """增强prompt流程"""
        # 批量生成增强prompt
        with torch.inference_mode():
            generated_responses = self._generate_responses(inputs)

        # 构建增强后的prompt
        enhanced_inputs = []
        inputs = [
            ex
            for ex in inputs
            for _ in range(self.num_generations)  # 每个输入重复 G 次
        ]


        for example, response in zip(inputs, generated_responses):
            description = response
            # 构建 result_str


            # 构建新的问题描述
            # new_problem = f"""Based on the following reasoning description, which refers to the region in the image corresponding to the target category "{example["target"]}": {response}, please identify and output the precise bounding box of the target area.The output must be in the following format:[x1, y1, x2, y2]"""
            new_problem = f"""The image is of resolution 1000×1000. Based on the following reasoning, which describes the region corresponding to the target category "{example["target"]}": {response}, please extract and output the exact bounding box coordinates of that region.Only return the result in the following format:[x1, y1, x2, y2]."""

            # 构建新的prompt
            new_prompt_content = [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": new_problem}
            ]
            new_prompt = [{"role": "user", "content": new_prompt_content}]

            # 创建新的示例
            new_example = example.copy()
            new_example["prompt"] = new_prompt
            new_example["problem"] = new_problem

            new_example["think"] = f"<think>{description}</think>"

            enhanced_inputs.append(new_example)

        return enhanced_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        inputs = self._enhance_prompts(inputs)

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        generation_config_2 = copy.deepcopy(self.generation_config)
        generation_config_2.num_return_sequences = 1
        generation_config_2.temperature = 0.7
        generation_config_2.repetition_penalty = 1.2

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=generation_config_2)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values,
                                                    image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask,
                                                                pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask,
                                                                    pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        """修改"""

        # 结构化处理生成结果
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        questions = []
        new_completions = []
        for ex, completion in zip(inputs, completions):
            think = ex['think']
            question = ex['problem']
            questions.append(question)
            completion = completion.replace('\n', '').replace('\r', '')
            new_completions.append([{"role": "assistant", "content": think + "<answer>" + completion + "</answer>"}])

        completions = new_completions


        # Compute the rewards
        # prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])

                output_reward_func = reward_func(prompts=prompts, completions=completions, step=self.step,
                                                 outputpath=self.args.output_dir, **reward_kwargs)

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        self.step += 1

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        # iou_now = metrics['rewards/accuracy_reward_iou']
        # print(iou_now)
        # if float(iou_now) > self.best_iou:
        #     print("大" * 100)
        #     self.best_iou = float(iou_now)
        #     if iou_now>0.499:
        #        self.model.save_pretrained(f"/data2/HSH/VRFT/Visual-RFT/code/bestmodel/model_{self.best_iou}")
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
