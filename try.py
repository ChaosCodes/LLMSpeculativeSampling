from transformers import AutoModel,LlamaForCausalLM, AutoTokenizer
import torch
from pathlib import Path
from typing import Optional
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/ByteLlama-320M-preview")
prompt = """Hi guys, I'm a undergrad from New York University"""
model = LlamaForCausalLM.from_pretrained("TinyLlama/ByteLlama-320M-preview").to("cuda")
# ByT5 Tokenizer will automatically append </s> to the end. Manually remove it here.
prompt_id = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")[:,:-1]
out = model.generate(prompt_id, max_length=1000, do_sample=True, num_return_sequences=1)
print(tokenizer.decode(out[0]))