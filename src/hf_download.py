import os
from huggingface_hub import snapshot_download

cache_root = "/home/tim/xai-hackathon-12-06-25/assets/huggingface"
os.makedirs(cache_root, exist_ok=True)

model_id = "Qwen/Qwen2.5-7B-Instruct"   # or "Qwen/Qwen2.5-14B-Instruct"

snapshot_download(repo_id=model_id, cache_dir=cache_root)
print("done")
