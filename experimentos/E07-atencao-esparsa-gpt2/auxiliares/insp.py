import transformers, inspect
from transformers.models.gpt2 import modeling_gpt2 as m
print(transformers.__version__)
print([n for n in dir(m.GPT2Attention) if 'attn' in n.lower() or n=='forward'])
print([n for n in dir(m) if 'attention' in n.lower()])
print(inspect.getsource(m.eager_attention_forward)[:1800])
print(inspect.getsource(m.GPT2Attention.forward)[-900:])
