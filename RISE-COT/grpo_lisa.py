
import shutil
import os
import re
import fcntl
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration,TrainerCallback

from math_verify import parse, verify

from trainer import Qwen2VLGRPOTrainer2, Qwen2VLGRPOVLLMTrainer,Qwen2VLGRPOTrainer_lisa
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import json
from typing import List, Union
import ast
import wandb
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




def extract_think_content(text: str) -> str:

    match = re.search(r'<think[^>]*>(.*?)</think>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_position_list(answer_str):

    try:
        match = re.search(r"\[\s*(-?\d+(?:\.\d+)?\s*,\s*){3}-?\d+(?:\.\d+)?\s*\]", answer_str)
        if match:
            position = ast.literal_eval(match.group(0))
            if isinstance(position, list) and len(position) == 4:
                return position
    except (ValueError, SyntaxError):
        pass
    return None

def bbox_to_center(bbox):

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_reward(gt_bbox, pred_bbox, image_size=(1000, 1000), alpha=0.4, beta=0.3, gamma=0.3):

    if gt_bbox is None or pred_bbox is None:
        return 0.0

    try:
        cx1, cy1, w1, h1 = bbox_to_center(gt_bbox)
        cx2, cy2, w2, h2 = bbox_to_center(pred_bbox)
    except Exception as e:
        print(f"[Warning] Invalid bbox for center conversion: {e}")
        return 0.0


    center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    if image_size:
        max_dist = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
        center_score = 1 - (center_dist / max_dist)
    else:
        center_score = np.exp(-center_dist / 100)


    aspect1 = w1 / (h1 + 1e-6)
    aspect2 = w2 / (h2 + 1e-6)
    if aspect1 <= 0 or aspect2 <= 0 or np.isnan(aspect1) or np.isnan(aspect2):
        aspect_ratio_score = 0.0
    else:
        aspect_ratio_score = 1 - abs(np.log(aspect1 / aspect2)) / np.log(10)
        aspect_ratio_score = np.clip(aspect_ratio_score, 0.0, 1.0)


    iou_score = compute_iou(gt_bbox, pred_bbox)
    reward = alpha * iou_score + beta * center_score + gamma * aspect_ratio_score
    reward = np.clip(reward, 0.0, 1.0)
    return iou_score



def safe_append_jsonl_line(file_path, data_list):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for data in data_list:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def accuracy_reward_iou(completions, solution,step, outputpath, image_path, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    data=[]
    for content, sol, img_path in zip(contents, solution, image_path):
        reward = 0.0

        student_answer_bbox = []
        ground_truth_bbox = []


        coord_pattern = r"^\s*[\[\(]\s*(-?\d+\.?\d*)\s*(,\s*(-?\d+\.?\d*)\s*)*[\]\)]\s*$"
        think = extract_think_content(content)

        ground_truth = sol.strip()
        content_match = re.search(r'<answer>(.*?)</answer>', content)
        student_answer = content_match.group(1).strip() if content_match else content.strip()

        ground_truth_bbox = extract_position_list(ground_truth)
        student_answer_bbox = extract_position_list(student_answer)

        reward = compute_reward(ground_truth_bbox, student_answer_bbox)

        if re.search(coord_pattern, think):
            reward = 0.1 * reward
        if reward > 1:
            reward = 1.0

        rewards.append(reward)
        data.append(
            {"img_path": img_path, "content": content, "solution": sol, "reward": reward, "pred_probs":student_answer_bbox,
             "gt_probs": ground_truth_bbox})


    os.makedirs(f"{outputpath}/reward", exist_ok=True)
    save_path = f"{outputpath}/reward/step_{step}.jsonl"

    safe_append_jsonl_line(save_path, data)

    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""

    pattern = r"<think>.*?</think>\s*<answer>\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]</answer>"

    completion_contents = [completion[0]["content"] for completion in completions]

    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    return [1.0 if match else 0.0 for match in matches]
def extract_info(text):

    pattern_count = r"Identified\s+(\d+)\s+potential\s+"
    match_count = re.search(pattern_count, text)
    if match_count:
        x = match_count.group(1)
    else:
        x = None


    pattern_targets = r"Target\s*(\d+)\s*:\s*(.+)"
    matches_targets = re.findall(pattern_targets, text)


    num_targets = len(matches_targets)


    targets = []
    for match in matches_targets:
        target_number = match[0]
        description = match[1].strip()
        targets.append(f"Target {target_number}: {description}")


    info = {
        "total_targets": x,
        "targets": targets,
        "num_targets": num_targets
    }

    return info




reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "format": format_reward,
}

SYSTEM_PROMPT = (

    ""
)


def main(script_args, training_args, model_args):

    script_args.reward_funcs = ['accuracy_iou','format']
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



    trainer_cls =Qwen2VLGRPOTrainer_lisa if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
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
        best_iou=0.0,

    )

    trainer.train()


    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)




if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
