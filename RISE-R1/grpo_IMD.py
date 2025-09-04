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
import numpy as np
import json

os.environ["WANDB_API_KEY"] = 'YOUR_API_KEY'
os.environ["WANDB_MODE"] = "offline"

class_list=[
    "laptop",
    "mouse",
    "backpack",
    "frying pan",
    "pencil sharpener",
    "digital clock",
    "paintbrush",
    "toilet tissue",
    "remote control",
    "bottlecap"
]


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

def pro_str_to_pro_list(prob_str, class_list):
    # Build regex pattern
    pattern = r'(\w+)["\']?\s*:\s*(\d+\.?\d*)'  # Match pattern like 'class':value
    matches = re.findall(pattern, prob_str)  # Find all matches

    # Initialize result dictionary
    result = {cls: 0 for cls in class_list}

    # Update result dictionary
    for match in matches:
        cls, prob = match
        if cls in result:
            try:
                prob = float(prob)  # Convert probability value to float
                if 0.0 <= prob <= 1.0:  # Check if probability value is between 0 and 1
                    result[cls] = prob
                else:
                    result[cls] = 0.0  # If probability value not between 0 and 1, set to 0
            except ValueError:
                # If probability value cannot be converted to float, set to 0
                result[cls] = 0.0
    return result

def extract_highest_probability(prob_str, class_list):
    """
    Extract the category with the highest probability from a string-formatted probability distribution

    Parameters:
        prob_str (str): Probability distribution string, format as "class1:prob1,class2:prob2,..."
        class_list (list): Category list to ensure extracted categories are in the list

    Returns:
        tuple: (Category with highest probability, Highest probability value)
    """
    result=pro_str_to_pro_list(prob_str,class_list)

    # Find category with highest probability and corresponding probability value
    highest_prob_category = max(result, key=result.get)
    highest_prob = result[highest_prob_category]

    return highest_prob_category, highest_prob




def kl_divergence(p, q):

    # Convert to NumPy array and copy to avoid in-place modification
    p = np.array(p, dtype=np.float64).copy()
    q = np.array(q, dtype=np.float64).copy()

    epsilon=1e-10

    # Numerical stability handling
    p = np.clip(p, epsilon, None)  # Only lower bound truncation, preserve original distribution shape
    q = np.clip(q, epsilon, None)  # Must strictly avoid q being zero

    p /= p.sum()
    q /= q.sum()

    return np.sum(p * np.log(p / q))


def hybrid_reward(gt_values, pred_values, alpha=0.5, beta=0.5):
    # Convert input to array and copy to avoid modifying original data
    p_kl = np.array(gt_values, dtype=np.float64).copy()
    q_kl = np.array(pred_values, dtype=np.float64).copy()

    epsilon = 1e-10
    # Numerical stability handling for KL part
    p_kl = np.clip(p_kl, epsilon, None)
    q_kl = np.clip(q_kl, epsilon, None)

    # Normalization to ensure probability sum is 1
    p_kl_normalized = p_kl / p_kl.sum()
    q_kl_normalized = q_kl / q_kl.sum()

    # Calculate KL divergence
    kl = np.sum(p_kl_normalized * np.log(p_kl_normalized / q_kl_normalized))

    # MSE calculation uses original input values (no normalization and clip processing)
    p_mse = np.array(gt_values, dtype=np.float64)
    q_mse = np.array(pred_values, dtype=np.float64)
    mse = np.mean((p_mse - q_mse) ** 2)

    # Calculate comprehensive reward
    reward = np.exp(-alpha * kl - beta * mse)
    return reward




def cos_mse_reward(gt_values, pred_values, alpha=0.8, beta=0.2,  epsilon=1e-10):
    """
    Calculate comprehensive reward combining cosine similarity and vector length prediction, with enhanced sensitivity to small probability values.

    Parameters:
    gt_values (array-like): Ground truth vector.
    pred_values (array-like): Predicted vector.
    alpha (float): Weight coefficient for cosine similarity.
    beta (float): Weight coefficient for vector length.
    epsilon (float): Small constant to prevent zero values in logarithmic calculations.

    Returns:
    float: Comprehensive reward value.
    """
    # Convert input to NumPy array
    gt = np.array(gt_values, dtype=np.float64)
    pred = np.array(pred_values, dtype=np.float64)

    # Calculate cosine similarity
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



def calculate_kl_reward_from_strings(pred_str, gt_str, class_list,beta=0.5):
    """
    Extract probability distributions from strings, calculate KL divergence and generate exponential decay reward.

    Parameters:
        pred_str (str): Predicted probability distribution string, format as "class1:prob1,class2:prob2,..."
        gt_str (str): Ground truth probability distribution string, same format
        class_list (list): List of all possible categories (should cover categories in pred_str and gt_str)
        beta (float): Parameter controlling reward decay rate (default 0.5)

    Returns:
        tuple: (reward, pred_probs_dict, gt_probs_dict)
               If calculation fails, reward=0.0

    Design notes:
        1. KL divergence direction is D_KL(Pred || GT), ensure GT distribution has no zero values
        2. Force probability normalization and numerical stability handling
    """
    # 1. Extract and validate probability distributions
    pred_probs = pro_str_to_pro_list(pred_str, class_list)
    gt_probs = pro_str_to_pro_list(gt_str, class_list)

    # 2. Ensure using complete category order defined by class_list
    keys = class_list  # Use class_list as baseline to avoid dependency on input order
    pred_values = [pred_probs.get(key, 0.0) for key in keys]
    gt_values = [gt_probs.get(key, 0.0) for key in keys]


    reward=hybrid_reward(gt_values,pred_values)
    return reward, pred_probs, gt_probs


def accuracy_reward(completions, solution, **kwargs):
    """Reward function with critical fixes applied."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []



    for content, sol in zip(contents, solution):
        reward = 0.0

        # String matching
        if reward == 0.0:
            try:
                think_match = re.search(r'<think>(.*?)</think>', content)
                think = think_match.group(1).strip() if think_match else content.strip()

                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                content_match = re.search(r'<answer>(.*?)</answer>', content)

                ground_truth = (sol_match.group(1) if sol_match else "").replace(' ', '').replace('_', '').lower()
                student_answer = (content_match.group(1) if content_match else "").replace(' ', '').replace('_', '').lower()


                reward,pred_probs,gt_probs=calculate_kl_reward_from_strings(student_answer,ground_truth,class_list)


                if re.search(r'\d+\.\d+', think):
                    reward *= 0.1  # Punishment deduction

            except (ValueError, re.error) as e:
                print(f"Verification error: {e}")

        rewards.append(reward)

    return rewards




def format_reward(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format.
    It also validates the key-value pairs in the <answer> section.

    Parameters:
        completions (list): List of completion dictionaries containing 'content'.
        class_list (list): List of valid class names.

    Returns:
        list: List of rewards (1.0 or 0.0) based on the format validity.
    """
    # Define regex pattern
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    kv_pattern = r'(\w+)["\']?\s*:\s*(\d+\.?\d*)'

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    rewards = []
    for match in matches:
        if match:
            # Extract content within <answer> tags
            answer_content = re.search(r"<answer>(.*?)</answer>", match.group(), re.DOTALL)
            if answer_content:
                answer_content = answer_content.group(1).strip()
                # Extract key-value pairs
                kv_pairs = re.findall(kv_pattern, answer_content)
                # Initialize result dictionary
                result_dict = {cls: 0 for cls in class_list}
                for key, value in kv_pairs:
                    if key in class_list:
                        try:
                            value = float(value)
                            if 0 <= value <= 1:
                                result_dict[key] = value
                            else:
                                # Value out of range, set to 0
                                result_dict[key] = 0
                        except ValueError:
                            # Invalid value format, set to 0
                            result_dict[key] = 0
                # Check if all key-value pairs are valid
                if all(value == 0 for value in result_dict.values()):
                    rewards.append(0.0)
                else:
                    offline = abs(sum(result_dict.values())-1)
                    rewards.append(np.exp(-offline))
            else:
                # No valid <answer> content found
                rewards.append(0.0)
        else:
            # No match for <think> and <answer> format
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
    # Get reward functions
    script_args.reward_funcs = ['accuracy', 'format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
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

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
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
