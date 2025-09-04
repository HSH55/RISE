
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoConfig

from math_verify import parse, verify

from trainer import Qwen2VLGRPOTrainer_class,Qwen2VLGRPOVLLMTrainer,Qwen2VLGRPOTrainer_class_multithink_emo6
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import fcntl
import json
import numpy as np
class_list=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
import os
import wandb
os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY'
os.environ["WANDB_MODE"] = "offline"


def pro_str_to_pro_list(prob_str, class_list):

    pattern = r'(\w+)["\']?\s*:\s*(\d+\.?\d*)'
    matches = re.findall(pattern, prob_str)


    result = {cls: 0 for cls in class_list}


    for match in matches:
        cls, prob = match
        if cls in result:
            try:
                prob = float(prob)
                if 0.0 <= prob <= 1.0:
                    result[cls] = prob
                else:
                    result[cls] = 0.0
            except ValueError:

                result[cls] = 0.0
    return result


def extract_highest_probability(prob_str, class_list):

    result=pro_str_to_pro_list(prob_str,class_list)

    highest_prob_category = max(result, key=result.get)
    highest_prob = result[highest_prob_category]

    return highest_prob_category, highest_prob




def kl_divergence(p, q):


    p = np.array(p, dtype=np.float64).copy()
    q = np.array(q, dtype=np.float64).copy()

    epsilon=1e-10


    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    p /= p.sum()
    q /= q.sum()

    return np.sum(p * np.log(p / q))


def hybrid_reward(gt_values, pred_values, alpha=0.5, beta=0.5):

    p_kl = np.array(gt_values, dtype=np.float64).copy()
    q_kl = np.array(pred_values, dtype=np.float64).copy()

    epsilon = 1e-10

    p_kl = np.clip(p_kl, epsilon, None)
    q_kl = np.clip(q_kl, epsilon, None)


    p_kl_normalized = p_kl / p_kl.sum()
    q_kl_normalized = q_kl / q_kl.sum()


    kl = np.sum(p_kl_normalized * np.log(p_kl_normalized / q_kl_normalized))


    p_mse = np.array(gt_values, dtype=np.float64)
    q_mse = np.array(pred_values, dtype=np.float64)
    mse = np.mean((p_mse - q_mse) ** 2)



    reward = np.exp(-alpha * kl - beta * mse)
    return reward



def cos_mse_reward(gt_values, pred_values, alpha=0.8, beta=0.2,  epsilon=1e-10):

    gt = np.array(gt_values, dtype=np.float64)
    pred = np.array(pred_values, dtype=np.float64)


    dot_product = np.dot(gt, pred)
    norm_gt = np.linalg.norm(gt)
    norm_pred = np.linalg.norm(pred)
    if norm_gt == 0 or norm_pred == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot_product / (norm_gt * norm_pred)

    cosine_similarity=abs(cosine_similarity)

    offline = np.exp(-np.dot(gt,abs(gt-pred)))
    reward =  cosine_similarity*alpha+offline*beta
    return reward



def calculate_kl_reward_from_strings(pred_str, gt_str, class_list, step,beta=0.5):

    pred_probs = pro_str_to_pro_list(pred_str, class_list)
    gt_probs = pro_str_to_pro_list(gt_str, class_list)


    keys = class_list
    pred_values = [pred_probs.get(key, 0.0) for key in keys]
    gt_values = [gt_probs.get(key, 0.0) for key in keys]


    reward=hybrid_reward(gt_values,pred_values)
    return reward, pred_probs, gt_probs



def safe_append_jsonl_line(file_path, data_list):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for data in data_list:
            json_line = json.dumps(data, ensure_ascii=False)
            f.write(json_line + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

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


def accuracy_reward(completions, solution, step, outputpath, image_path, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    data = []



    for content, sol, img_path in zip(contents, solution, image_path):
        reward = 0.0

        if reward == 0.0:
            try:
                think_match = re.search(r'<think>(.*?)</think>', content)
                think = think_match.group(1).strip() if think_match else content.strip()

                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                content_match = re.search(r'<answer>(.*?)</answer>', content)

                ground_truth = (sol_match.group(1) if sol_match else "").replace(' ', '').replace('_', '').lower()
                student_answer = (content_match.group(1) if content_match else "").replace(' ', '').replace('_', '').lower()


                reward,pred_probs,gt_probs=calculate_kl_reward_from_strings(student_answer,ground_truth,class_list,step)



                if re.search(r'\d+\.\d+', think):
                    reward *= 0.1

            except (ValueError, re.error) as e:
                print(f"Verification error: {e}")

        rewards.append(reward)
        data.append({"img_path":img_path,"content": content, "solution": sol, "reward": reward, "pred_probs": pred_probs, "gt_probs": gt_probs})



    os.makedirs(f"{outputpath}/reward", exist_ok=True)
    save_path = f"{outputpath}/reward/step_{step}.jsonl"

    safe_append_jsonl_line(save_path,data)


    return rewards




def format_reward(completions, **kwargs):

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    kv_pattern = r'(\w+)["\']?\s*:\s*(\d+\.?\d*)'

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    rewards = []
    for match in matches:
        if match:

            answer_content = re.search(r"<answer>(.*?)</answer>", match.group(), re.DOTALL)
            if answer_content:
                answer_content = answer_content.group(1).strip()

                kv_pairs = re.findall(kv_pattern, answer_content)

                result_dict = {cls: 0 for cls in class_list}
                for key, value in kv_pairs:
                    if key in class_list:
                        try:
                            value = float(value)
                            if 0 <= value <= 1:
                                result_dict[key] = value
                            else:

                                result_dict[key] = 0
                        except ValueError:

                            result_dict[key] = 0

                if all(value == 0 for value in result_dict.values()):
                    rewards.append(0.0)
                else:

                    offline = abs(sum(result_dict.values())-1)
                    rewards.append(np.exp(-offline))
            else:

                rewards.append(0.0)
        else:

            rewards.append(0.0)

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
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


    trainer_cls = Qwen2VLGRPOTrainer_class_multithink_emo6 if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
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
