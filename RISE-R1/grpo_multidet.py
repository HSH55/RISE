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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import re
import json
os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY'
os.environ["WANDB_MODE"] = "offline"
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    lora_path: Optional[str] = field(default=None)


def convert_bbox_list(bboxes):


    converted = []
    for bbox in bboxes:
        if len(bbox) != 4:
            continue

        try:

            x, y, w, h = map(float, bbox)

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # 添加到结果列表
            converted.append([x1, y1, x2, y2])

        except (ValueError, TypeError):
            print(f"ERROR 311")

    return converted

def extract_bbox(response):

    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_matches:
        return None

    content = answer_matches[-1].strip()

    bbox_matches = re.findall(r'\[[^\[\]]+\]', content)

    bbox_list = []
    for match in bbox_matches:
        try:

            clean_match = match.replace('\n', '').replace(' ', '')


            bbox = json.loads(clean_match)


            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                bbox_list.append(bbox)
        except (json.JSONDecodeError, TypeError):

            try:

                numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', match)
                if len(numbers) == 4:
                    bbox = [float(num) for num in numbers]
                    bbox_list.append(bbox)
            except:
                continue

    return bbox_list if bbox_list else None


def multi_bbox_iou(gt_boxes, pred_boxes):


    if not pred_boxes or len(pred_boxes) == 0:
        return 0.0

    if not gt_boxes or len(gt_boxes) == 0:
        return 0.0


    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt_box, pred_box)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # 负号因为我们想要最大化IOU

    matched_iou_sum = 0.0

    for i, j in zip(row_ind, col_ind):
        matched_iou_sum += iou_matrix[i, j]

    average_iou = matched_iou_sum / len(pred_boxes)

    return average_iou


def calculate_iou(bbox1, bbox2):


    xi1 = max(bbox1[0], bbox2[0])
    yi1 = max(bbox1[1], bbox2[1])
    xi2 = min(bbox1[2], bbox2[2])
    yi2 = min(bbox1[3], bbox2[3])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection_area = (xi2 - xi1) * (yi2 - yi1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def accuracy_reward_iou(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        reward = 0.0
        debug_info = {
            "content": content[:100] + "..." if len(content) > 100 else content,
            "sol": sol[:100] + "..." if len(sol) > 100 else sol,
            "gt_bbox": None,
            "pred_bbox": None,
            "iou": None,
            "error": None
        }

        try:

            ground_truth_bbox = extract_bbox(sol)
            ground_truth_bbox=convert_bbox_list(ground_truth_bbox)
            student_answer_bbox = extract_bbox(content)

            debug_info["gt_bbox"] = ground_truth_bbox
            debug_info["pred_bbox"] = student_answer_bbox

            if not ground_truth_bbox:
                debug_info["error"] = "No ground truth bbox found"
                reward = 0.0
            elif not student_answer_bbox:
                debug_info["error"] = "No predicted bbox found"
                reward = 0.0
            else:

                average_iou = multi_bbox_iou(ground_truth_bbox, student_answer_bbox)
                reward = min(max(average_iou, 0.0), 1.0)  # 确保在[0,1]范围内
                debug_info["iou"] = average_iou

        except Exception as e:
            debug_info["error"] = str(e)

        debug_info["reward"] = reward
        rewards.append(reward)
        print(f"DEBUG: {debug_info}")

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_iou','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")


    trainer_cls =Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].select(range(16)),
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        best_iou=0.0,
        # callbacks=[SAVEMODELCallback()],
    )
    #
    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)




if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
