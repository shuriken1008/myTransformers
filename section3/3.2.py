from transformers import AutoTokenizer
from bertviz.transformers_neuron_view import BertModel
from bertviz.neuron_view import show


inputs_embeds = token_emb(inputs.input_ids)
inputs_embeds.size()


import torch
from math import sqrt
query = key = value = inputs_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)
scores.size()


#ソフトマックスの適用

import torch.nn.functional as F
weights = F.softmax(scores, dim=-1)
weights.sum(dim=-1)

attn_outputs = torch.bmm(weights, value)
attn_outputs.shape

def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


