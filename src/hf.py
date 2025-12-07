import json, random, os
import torch
import pandas as pd
from tqdm import tqdm
import transformers

from constants import *
from power_samp_utils import format_prompt, AutoregressiveSampler, mcmc_power_samp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cache_dir = "/home/tim/xai-hackathon-12-06-25/assets/huggingface"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_str = "Qwen/Qwen2.5-7B"

repo_root = os.environ.get("REPO_ROOT", os.getcwd())
json_file = os.path.join(repo_root, "assets", "data", "MATH500.json")

with open(json_file) as f:
    dataset = json.load(f)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, cache_dir=cache_dir, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_str, dtype="auto", device_map="auto", trust_remote_code=True
)
autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

def run_qwen(dataset, start=0, end=100, cot=True, temp=0.25, mcmc_steps=10, max_new_tokens=128, seed=0):
    random.seed(seed)
    results = []
    for _, data in tqdm(list(enumerate(dataset[start:end])), desc="MATH"):
        res = {}
        question = data["prompt"]
        answer = data["answer"]
        res["question"] = question
        res["answer"] = answer

        input_text = format_prompt(question, "qwen", tokenizer, cot)
        inputs = tokenizer(input_text, return_tensors="pt", padding=False, truncation=False)
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        mcmc_ids_, _, _, acc = mcmc_power_samp(
            autoreg_sampler, inputs["input_ids"][0].tolist(), temp, mcmc_steps, max_new_tokens=max_new_tokens
        )
        mcmc_ids = torch.tensor(mcmc_ids_, dtype=torch.long, device="cpu")
        mcmc_completion = tokenizer.decode(mcmc_ids, skip_special_tokens=True)
        res["mcmc_completion"] = mcmc_completion
        results.append(res)
        return res

    return pd.DataFrame(results)


res = run_qwen(dataset, start=0, end=1, cot=True, temp=0.25, max_new_tokens=1<<8, mcmc_steps=1)

print(res)