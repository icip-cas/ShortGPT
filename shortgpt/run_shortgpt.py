from tqdm.notebook import tqdm

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from shortgpt import ShortGPT

data = load_dataset("emozilla/pg19", split="validation")
dataloader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
    pin_memory=True,
    generator=torch.Generator()
)

short_model = ShortGPT(
    model_name="meta-llama/Llama-2-7b",
    layers_path="model.layers",
    n_prune_layers=9
)

for batch in dataloader:
    prompts = batch['text']

    short_model.eval_importance(
        prompts=prompts,
        stride=256,
        max_gen_len=0
    )

prune_order = [i for i, _ in sorted(enumerate(short_model.importances), key=lambda x: x[1])]
print("Pruning order:", ', '.join(str(i) for i in prune_order))

short_model.remove_layers()

for layer_idx, module in enumerate(short_model.layers):
    module.self_attn.layer_idx = layer_idx