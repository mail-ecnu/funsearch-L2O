from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = snapshot_download("qwen/Qwen1.5-1.8B")
# Loading local checkpoints
# trust_remote_code is still set as True since we still load codes from local dir instead of transformers
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()