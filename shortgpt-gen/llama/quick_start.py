import torch
from transformers import LlamaTokenizer
from shortgpt_llama import ShortgptLlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b", use_fast=False, trust_remote_code=True)
model = ShortgptLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b", device_map="auto")
model.model.prune_layers = [21,22,23,24,25,26,27,28,29]
inputs = tokenizer('I believe the meaning of life is', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
