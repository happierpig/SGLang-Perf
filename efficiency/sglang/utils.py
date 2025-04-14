import random

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_system_prompt(model_name: str):
    # Prompt Suggested by
    # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    if "llama" in model_name or "Llama" in model_name:
        prompt = (
            "<|im_start|>user\n"
            "{}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif "qwen" in model_name or "Qwen" in model_name:
        prompt = (
            "<|im_start|>user\n"
            "{}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        raise ValueError("Invalid model name")

    return prompt


def get_dataset(
    model_name: str,
    dataset: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = -1,
):
    """
    Load the dataset based on the provided dataset name and number of requests.
    If num_requests is greater than 0, sample that many requests from the dataset.
    """
    system_prompt = get_system_prompt(model_name)

    if dataset == "aime":
        dataset = load_dataset("AI-MO/aimo-validation-aime")
        problems = dataset["train"]
        problems = [problem["problem"] for problem in problems]
    elif dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        problems = dataset["test"]
        problems = [problem["problem"] for problem in problems]
    elif dataset == "gpqa-diamond":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        problems = dataset["train"]
        problems = [problem["Question"] for problem in problems]
    else:
        raise ValueError("Invalid dataset name")

    if num_requests > 0:
        # setup random seed for reproducibility
        random.seed(42)
        problems = random.choices(problems, k=num_requests)

    problems = [system_prompt.format(problem) for problem in problems]

    ret_dataset = []
    for p in problems:
        prompt_token_ids = tokenizer.encode(p)
        # maximal output length: 112k
        # to avoid SGLang maximal model length check
        ret_dataset.append((p, len(prompt_token_ids), 112 * 1024))

    return ret_dataset


# Example usage:
# tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# print(get_dataset("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "aime", tok, 4))

# system_prompt = get_system_prompt("qwen")
# print(system_prompt.format("What is 1+1?"))
