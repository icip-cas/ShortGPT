# ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
<p align="center">
  📄 <a href="https://arxiv.org/abs/2403.03853" target="_blank">Paper</a> 
</p>

## ShortGPT: Identifying and Removing Redundant Layers

```bash
python shortgpt/run_shortgpt.py
```

You will get the order of layer pruning and the pruned model for Llama2-7B.
For other models, please replace model_name in line 19 of shortgpt/run_shortgpt.py

The layer ids of pruned layers are provided below.

| Model | Checkpoint | Pruned Layers |
| --- | --- | --- |
| Llama2-7B | [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) |  21,22,23,24,25,26,27,28,29 |
| Llama2-13B | [meta-llama/Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b) | 26,27,28,29,30,31,32,33,34,35 |
| Baichuan2-7B | [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) |  22,23,24,25,26,27,28,29,30 |
| Baichuan2-13B | [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base) |  26,27,28,29,30,31,32,33,34,35 |

## ShortGPT-gen for Generative Tasks

### Llama Models

```bash
python shortgpt-gen/llama/quick_start.py
```

### Baichuan2 Models

```bash
python shortgpt-gen/baichuan/quick_start.py
```

## Citation
If you find ShortGPT useful for your research and applications, please cite using this BibTeX:
```bib
@article{men2024shortgpt,
  title={Shortgpt: Layers in large language models are more redundant than you expect},
  author={Men, Xin and Xu, Mingyu and Zhang, Qingyu and Wang, Bingning and Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Chen, Weipeng},
  journal={arXiv preprint arXiv:2403.03853},
  year={2024}
}
```