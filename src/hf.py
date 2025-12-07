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
json_file = "../assets/data/MATH500.json"

with open(json_file, "r") as f:
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
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"][0]

        naive_out = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        naive_out_temp = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
        )

        # naive_ids = naive_out[0][:, len(input_ids):].squeeze().to("cpu")
        # naive_completion = tokenizer.decode(naive_ids, skip_special_tokens=True)
        # res["naive_completion"] = naive_completion
        # res["naive_answer"] = parse_answer(naive_completion)

        # naive_temp_ids = naive_out_temp[0][:, len(input_ids):].squeeze().to("cpu")
        # naive_temp_completion = tokenizer.decode(naive_temp_ids, skip_special_tokens=True)
        # res["naive_temp_completion"] = naive_temp_completion
        # res["naive_temp_answer"] = parse_answer(naive_temp_completion)
        
        mcmc_ids, _, _, acc = mcmc_power_samp(
            autoreg_sampler, input_ids.tolist(), temp, mcmc_steps, max_new_tokens=max_new_tokens
        )
        mcmc_ids = mcmc_ids.to("cpu")
        mcmc_completion = tokenizer.decode(mcmc_ids, skip_special_tokens=True)
        res["mcmc_completion"] = mcmc_completion
        results.append(res)
        return res

    return pd.DataFrame(results)


res = run_qwen(dataset, start=0, end=1, cot=True, temp=0.25, max_new_tokens=1<<10, mcmc_steps=1)

print(res)