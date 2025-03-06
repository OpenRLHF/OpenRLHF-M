import random
import re
import torch

# pip install latex2sympy2_extended math_verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


prompt_template = "chatml"
format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

if prompt_template == "chatml":
    problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
    response_prefix = r"<\|im_start\|>assistant\n"
elif prompt_template == "qwen1":
    problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
    response_prefix = r"<｜Assistant｜>"
elif prompt_template == "base":
    problem_pattern = r"User: (.*?)\n\nAssistant:"
    response_prefix = r"Assistant: "


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1


def verify_math(content, sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            reward = 0
            print("Failed to verify: ", e)
    else:
        reward = 0
        print("Failed to parse gold solution: ", sol)
    return reward


def reward_func(queries, prompts, labels):
    rewards = []
    for q, problem, answer in zip(queries, prompts, labels):
        response = get_response_from_query(q) or q
        format_reward = float(verify_format(response)) * 0.2
        acc_reward = float(verify_math(response, answer))
        rewards.append(format_reward + acc_reward)

        do_print = random.randint(1, 20) == 1
        if do_print:
            info = f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\n"
            info = re.sub(r"<\|.*?\|>", "", info)
            print(info)

    return torch.tensor(rewards)
