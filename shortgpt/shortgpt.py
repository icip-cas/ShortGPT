from typing import List, Optional

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from metrics import *

class ShortGPT():

    def __init__(self, model_name: str, layers_path: str, n_prune_layers: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

        modules = layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod

        self.n_prune_layers = n_prune_layers
        self.importances = [0 for _ in self.layers]

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = []
    ):
        layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        for layer_idx in sorted(layers_to_remove, reverse=True):
            del self.layers[layer_idx]
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor]):
        n = 1

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9
    ):
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids
            inputs = input_ids[seq_ids, start:start+stride]
            attn = attn_mask[seq_ids, start:start+stride]

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            
            self.compute_bi(outputs.hidden_states)

        return
