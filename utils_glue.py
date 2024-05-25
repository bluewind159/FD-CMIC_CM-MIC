import torch
import copy
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# kd loss
def cal_loss(s_logits, t_logits, temperature):
    soft_labels = F.log_softmax(t_logits / temperature, dim=-1, dtype=torch.float32)
    log_prob = F.log_softmax(s_logits / temperature, dim=-1, dtype=torch.float32)
    ori_kld_loss = -torch.exp(soft_labels) * log_prob + torch.exp(soft_labels) * soft_labels
    loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))
    return loss
    
def cal_js_loss(s_logits, t_logits, temperature):
    m_logits=0.5 * (s_logits+t_logits)
    kl1 = cal_loss(m_logits,s_logits,temperature) # kl(y_pred||M)
    kl2 = cal_loss(m_logits,t_logits,temperature) # kl(y_true||M)
    kjs = (kl1+kl2)/2
    return kjs