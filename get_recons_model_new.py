import torch
import torch.nn as nn
from torch.nn.functional import softplus
from transformers import AdamW
import numpy as np
import torch.nn.functional as F
import os
import random
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import copy
import distributed as dist_fn

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, labels):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        #_, embed_ind = (-dist).max(1)
        embed_ind = labels
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        #quantize = quantize*labels.unsqueeze(-1)
        #input = input*labels.unsqueeze(-1)
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        #quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
class recons(nn.Module):
    def __init__(self, size1):
        super(recons, self).__init__()
        self.recons=nn.Linear(768+1,768)
        self.act = ACT2FN["gelu"]
        self.dropout=nn.Dropout(0.1)
            
    def forward(self, x, labels):
        labels=labels.unsqueeze(-1)
        result=self.recons(torch.cat([x,labels],dim=-1))
        result=self.act(result)
        return result
        
class proto_recons(nn.Module):
    def __init__(self, size1):
        super(proto_recons, self).__init__()
        self.recons=nn.Linear(768*2,768)
        self.quant= Quantize(768,2)
        self.act = ACT2FN["gelu"]
        self.dropout=nn.Dropout(0.1)
            
    def forward(self, x, labels, x_pos_trans=None, x_pooled_output=None):
        if x_pos_trans is not None:
            result=self.recons(torch.cat([x,x_pos_trans],dim=-1))
            result=self.act(result)
            return result
        else:
            x_pos_trans, diff, embed_ind = self.quant(x_pooled_output, labels)
            result=self.recons(torch.cat([x,x_pos_trans],dim=-1))
            result=self.act(result)
            return result, diff

def get_recons_model(args=None,size1=768):
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    print('ReLU activation')
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=recons(size1)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return model.cuda(),optimizer
    
def get_proto_recons_model(args=None,size1=768):
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    print('ReLU activation')
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model=proto_recons(size1)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return model.cuda(),optimizer


if __name__ == '__main__':
    main()
