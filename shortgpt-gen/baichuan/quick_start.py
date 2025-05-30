import torch
from transformers import AutoTokenizer
from shortgpt_baichuan import ShortgptBaichuanForCausalLM
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
model = ShortgptBaichuanForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", device_map="auto")
model.model.prune_layers = [22,23,24,25,26,27,28,29,30]
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
