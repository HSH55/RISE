
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
import fcntl
import json
import datetime
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
from deepspeed import zero

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
class_list=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

def remove_sentences_with_word_and_clean_newlines(text, words):
    """
    Remove sentences containing specific words and clean up extra newlines.

    Args:
        text (str): Original text
        words (list): List of words to remove

    Returns:
        str: Text with sentences containing specified words removed and cleaned
    """
    text_without_newlines = text.replace('\n', ' ')
    cleaned_text = text_without_newlines
    for word in words:
        pattern = re.compile(r'[^。！？.!?]*' + re.escape(word) + r'[^。！？.!?]*[。！？.!?]')
        cleaned_text = re.sub(pattern, '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def compare_zero3_models(model_a, model_b):
    """Safely compare Zero3 sharded model parameters"""
    if model_a is None or model_b is None:
        raise ValueError("Model objects cannot be None")

    with zero.GatheredParameters(list(model_a.parameters()), modifier_rank=0) as a_params, \
            zero.GatheredParameters(list(model_b.parameters()), modifier_rank=0) as b_params:

        if torch.distributed.get_rank() == 0:
            if a_params is None or b_params is None:
                print("⚠️ Parameter collection failed, models may not be properly initialized")
                return False

            a_params = [p.cpu().detach() for p in a_params if p is not None]
            b_params = [p.cpu().detach() for p in b_params if p is not None]

            if len(a_params) != len(b_params):
                return False

            return all(torch.allclose(p1, p2, rtol=1e-3, atol=1e-5)
                       for p1, p2 in zip(a_params, b_params))
        else:
            return False

def safe_write_tokens(TOKENfile_path, per_token_logps, ref_per_token_logps, per_token_kl):
    """
    Multi-GPU safe writing of token-level data

    Args:
        TOKENfile_path: Log file path
        per_token_logps: Current model token log probabilities (tensor)
        ref_per_token_logps: Reference model token log probabilities (tensor)
        per_token_kl: Computed KL divergence (tensor)
    """

    with open(TOKENfile_path, 'a+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            f.write(f"per_token_logps:\n{per_token_logps}\n")
            f.write(f"ref_token_logps:\n{ref_per_token_logps}\n")
            f.write(f"per_token_kl:\n{per_token_kl}\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def extract_first_class(processed_comp, class_list):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(cls) for cls in class_list) + r')\b', flags=re.IGNORECASE)
    match = pattern.search(processed_comp)
    if match:
        first_class = match.group(0)
        return f"<answer>{first_class}</answer>"
    else:
        return ""

def extract_valid_probabilities(prob_str, class_list):
    """
    Extract valid key-value pairs from string and form final answer with class_list

    Args:
        prob_str (str): Probability distribution string in format "{class1:prob1,class2:prob2,...}"
        class_list (list): Class list to ensure extracted classes are valid

    Returns:
        dict: Final answer dictionary with classes as keys and valid probability values
    """
    pattern = r'(\w+)["\']?\s*:\s*(\d+\.?\d*)'
    matches = re.findall(pattern, prob_str)
    result = {cls: 0 for cls in class_list}

    for match in matches:
        cls, prob = match
        if cls in result:
            try:
                prob = float(prob)
                if 0 <= prob <= 1:
                    result[cls] = prob
                else:
                    result[cls] = 0
            except ValueError:
                result[cls] = 0

    return result

def remove_class_names(text, class_list):
    """
    Remove specific class names from text
    """
    pattern = re.compile(r'\b(' + '|'.join(re.escape(cls) for cls in class_list) + r')\b', flags=re.IGNORECASE)
    cleaned = re.sub(pattern, '', text)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()

def process_text(text):
    processed_text = re.sub(r'\n\s*\n', '\n\n', text)
    processed_text = re.sub(r'^\s+', '', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'\s+$', '', processed_text, flags=re.MULTILINE)
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
    processed_text = processed_text.replace('\n', '').replace('\r', '')
    return processed_text

def save_qa_json(questions, completions, step_number, save_dir):
    """
    Save question-answer pairs in JSON format

    Args:
        questions (list): Original question list
        completions (list): Corresponding model answer list
        step_number (int): Current processing step number
        save_dir (str): File storage path
    """
    save_dir = save_dir + "/qa_result"
    if len(questions) != len(completions):
        raise ValueError(f"Question count ({len(questions)}) doesn't match answer count ({len(completions)})")

    os.makedirs(save_dir, exist_ok=True)

    save_data = {
        "metadata": {
            "step": step_number,
            "timestamp": datetime.datetime.now().isoformat(),
            "data_version": "1.1"
        },
        "qa_pairs": [
            {
                "question_id": idx,
                "question_text": str(q),
                "model_answer": str(a)
            }
            for idx, (q, a) in enumerate(zip(questions, completions))
        ]
    }

    file_path = os.path.join(save_dir, f"step_{step_number}_qa.json")

    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"JSON results saved to: {file_path}")

class Qwen2VLGRPOTrainer_class_multithink_emo6(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
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
        self.step = step
        self.best_iou = best_iou

        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs
                )
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
            print('Using LoRA')
            model = get_peft_model(model, peft_config)

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
                print("DeepSpeed not properly used")
                self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

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

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

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
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True
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
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                       image_grid_thw=image_grid_thw).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _generate_responses(self, inputs):
        if self.step <= -10:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "/data2/HSH/models--Qwen--Qwen2.5-VL-72B-Instruct/model",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            processor = AutoProcessor.from_pretrained("/data2/HSH/models--Qwen--Qwen2.5-VL-72B-Instruct/model",
                                                      min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            model = self.model
            processor = self.processing_class
        device = model.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, processor)["prompt"].replace("this distribution.","this distribution concisely in under 100 words.") for example in inputs]
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

        for key in prompt_inputs.keys():
            if isinstance(prompt_inputs[key], torch.Tensor):
                prompt_inputs[key] = prompt_inputs[key].to(device)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        generation_config_1 = copy.deepcopy(self.generation_config)
        generation_config_1.num_return_sequences=8

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                generation_config=generation_config_1
            )

        completions = processor.batch_decode(prompt_completion_ids, skip_special_tokens=True)

        cleaned_responses = [
            process_text(self.extract_assistant_content(text))
            for text in completions
        ]
        print(f"Model output: {cleaned_responses}")

        return cleaned_responses

    def extract_assistant_content(self, text):
        match = re.search(r'assistant\n(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return None

    def _enhance_prompts(self, inputs):
        with torch.inference_mode():
            generated_responses = self._generate_responses(inputs)

        enhanced_inputs = []
        inputs = [
            ex
            for ex in inputs
            for _ in range(self.num_generations)
        ]

        for example, response in zip(inputs, generated_responses):
            description = remove_sentences_with_word_and_clean_newlines(response,class_list)
            Categories = "'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'"
            result_str = "{'anger': 0.0, 'disgust': 0.1666667, 'fear': 0.1, 'joy': 0.466667, 'sadness': 0.1, 'surprise': 0.166667, 'neutral': 0.0}"

            new_problem = f"""Please analyze the provided image description and generate a probability distribution across the following emotion categories: {Categories}. Description: {description}. Do not provide any additional explanations or reasoning. Only return the result in the specified format. Each probability must be a floating-point number between 0 and 1, rounded to 6 decimal places. The sum of all probabilities must be exactly 1.000000 (allowing for minor floating-point errors). Example output: {result_str}. Incorrect output: {{'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 1.0, 'sadness': 0, 'surprise': 0, 'neutral': 0}}."""

            new_prompt_content = [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": new_problem}
            ]
            new_prompt = [{"role": "user", "content": new_prompt_content}]

            new_example = example.copy()
            new_example["prompt"] = new_prompt
            new_example["problem"] = new_problem
            new_example["think"] = f"<think>{description}</think>"

            enhanced_inputs.append(new_example)

        return enhanced_inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        for key in prompt_inputs.keys():
            if isinstance(prompt_inputs[key], torch.Tensor):
                prompt_inputs[key] = prompt_inputs[key].to('cuda')

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

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

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        print(completions)

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

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
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, step=self.step,
                                                 outputpath=self.args.output_dir, **reward_kwargs)

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

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

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
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
