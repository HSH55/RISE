
import os
import re
import fcntl
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify

from trainer import Qwen2VLGRPOTrainer_coco, Qwen2VLGRPOVLLMTrainer
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


def convert_bbox_list(bboxes):

    converted = []
    for bbox in bboxes:

        if len(bbox) != 4:
            print(f"警告: 跳过无效边界框 {bbox} - 需要4个值")
            continue

        try:

            x, y, w, h = map(float, bbox)


            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h


            converted.append([x1, y1, x2, y2])

        except (ValueError, TypeError):
            print(f"警告: 无法转换边界框 {bbox} - 值必须是数字")

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


    row_ind, col_ind = linear_sum_assignment(-iou_matrix)


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

def extract_think_content(text: str) -> str:

    match = re.search(r'<think[^>]*>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def accuracy_reward_iou(completions, solution, step, outputpath, image_path, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    data = []

    for content, sol, img_path,target in zip(contents, solution, image_path,target):
        reward = 0.0
        student_answer_bbox = []
        ground_truth_bbox = []
        debug_info = {
            "img_path": img_path,
            "content": content,
            "sol": sol,
            "target":target,
            "gt_bbox": None,
            "pred_bbox": None,
            "reward": None,
            "error": None,
        }

        """检测坐标泄露"""
        coord_pattern = r"^\s*[\[\(]\s*(-?\d+\.?\d*)\s*(,\s*(-?\d+\.?\d*)\s*)*[\]\)]\s*$"
        think = extract_think_content(content)

        ground_truth_bbox = extract_bbox(sol)
        student_answer_bbox = extract_bbox(content)

        debug_info["gt_bbox"] = ground_truth_bbox
        debug_info["pred_bbox"] = student_answer_bbox


        average_iou = multi_bbox_iou(ground_truth_bbox, student_answer_bbox)
        reward = min(max(average_iou, 0.0), 1.0)
        debug_info["iou"] = average_iou


        if re.search(coord_pattern, think):
            reward *= 0.1
        reward = min(reward, 1.0)



        debug_info["reward"] = reward
        debug_info["gt_bbox"] = ground_truth_bbox
        debug_info["pred_bbox"] = student_answer_bbox
        rewards.append(reward)
        data.append(debug_info)


    os.makedirs(f"{outputpath}/reward", exist_ok=True)
    save_path = f"{outputpath}/reward/step_{step}.jsonl"

    safe_append_jsonl_line(save_path, data)

    return rewards


def safe_append_jsonl_line(file_path, data_list):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for data in data_list:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]

    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward_iou,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):

    script_args.reward_funcs = ['accuracy', 'format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]



    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk(script_args.dataset_name)


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
        dataset = dataset.map(make_conversation_image)

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")


    trainer_cls = Qwen2VLGRPOTrainer_coco if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )


    if training_args.resume_from_checkpoint:

        if not os.path.exists(training_args.resume_from_checkpoint):
            raise ValueError(f"Checkpoint path {training_args.resume_from_checkpoint} does not exist.")


        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:

        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

