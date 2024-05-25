import torch
import torch.nn as nn
from torch.nn.functional import softplus
from transformers import AdamW
import numpy as np
import torch.nn.functional as F
import os
import random
from transformers.activations import ACT2FN
from utils_glue import cal_loss
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
class model_trans(nn.Module):
    def __init__(self, size1, size2):
        super(model_trans, self).__init__()
        self.pos_trans = nn.Linear(size1, size2)
        self.neg_trans = nn.Linear(size1, size2)
        #self.act = nn.ReLU()
        self.act = ACT2FN["gelu"]
        self.dropout=nn.Dropout(0.1)
        self.use_memory=False
    def forward(self, x1):
        pos_result=self.act(self.pos_trans(x1))
        pos_result=self.dropout(pos_result)
        neg_result=self.act(self.neg_trans(x1))
        neg_result=self.dropout(neg_result)
        return pos_result, neg_result
        
class model_split(nn.Module):
    def __init__(self, size1=768, size2=768,num=6):
        super(model_split, self).__init__()
        self.compress=nn.ModuleList()
        self.expanddd=nn.ModuleList()
        self.reversee=nn.ModuleList()
        size_sp=[size1-128*i for i in range(num)]
        for i in range(num-1):
            self.comp = nn.Sequential(nn.Linear(size_sp[i], size_sp[i+1]),
                                      ACT2FN["gelu"])
            self.expa = nn.Sequential(nn.Linear(size_sp[i+1], size_sp[i]),
                                      ACT2FN["gelu"])
            self.reve = nn.Sequential(nn.Linear(size_sp[i+1]+size_sp[i], size2),
                                      ACT2FN["gelu"])
            self.compress.append(self.comp)
            self.expanddd.append(self.expa)
            self.reversee.append(self.reve)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self, x1):
        result=[]
        fusion_f=x1
        for layer_c, layer_e, layer_r in zip(self.compress,self.expanddd,self.reversee):
            comp_f=layer_c(fusion_f)
            #expa_f=layer_e(comp_f)
            reve_f=layer_r(torch.cat([comp_f,fusion_f],dim=-1))
            fusion_f=comp_f
            result.append(self.dropout(reve_f))
        return result
        
class model_U_split(nn.Module):
    def __init__(self, size1=768, size2=768,num=6):
        super(model_U_split, self).__init__()
        print('U split model!!!!!!!!!!!!!!!!!!!!!')
        size_sp=[size1-128*(i+1) for i in range(num-1)]
        self.comp0_0 = nn.Sequential(nn.Linear(size1, size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.comp1_0 = nn.Sequential(nn.Linear(size_sp[0], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.comp2_0 = nn.Sequential(nn.Linear(size_sp[1], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.comp3_0 = nn.Sequential(nn.Linear(size_sp[2], size_sp[3]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.comp4_0 = nn.Sequential(nn.Linear(size_sp[3], size_sp[4]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
        self.expa3_1 = nn.Sequential(nn.Linear(size_sp[3]+size_sp[4], size_sp[3]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.expa2_2 = nn.Sequential(nn.Linear(size_sp[2]+size_sp[3], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.expa1_3 = nn.Sequential(nn.Linear(size_sp[1]+size_sp[2], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.expa0_4 = nn.Sequential(nn.Linear(size_sp[0]+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
        self.reve1 = nn.Sequential(nn.Linear(size_sp[0], size2), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.reve2 = nn.Sequential(nn.Linear(size_sp[1], size2), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.reve3 = nn.Sequential(nn.Linear(size_sp[2], size2), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.reve4 = nn.Sequential(nn.Linear(size_sp[3], size2), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.reve5 = nn.Sequential(nn.Linear(size_sp[4], size2), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
    def forward(self, x1):
        x0_0=self.comp0_0(x1)
        x1_0=self.comp1_0(x0_0)
        x2_0=self.comp2_0(x1_0)
        x3_0=self.comp3_0(x2_0)
        x4_0=self.comp4_0(x3_0)
        
        x3_1=self.expa3_1(torch.cat([x3_0,x4_0],dim=-1))
        x2_2=self.expa2_2(torch.cat([x2_0,x3_1],dim=-1))
        x1_3=self.expa1_3(torch.cat([x1_0,x2_2],dim=-1))
        x0_4=self.expa0_4(torch.cat([x0_0,x1_3],dim=-1))
        
        result=[]
        result.append(self.reve1(x0_4))
        result.append(self.reve2(x1_3))
        result.append(self.reve3(x2_2))
        result.append(self.reve4(x3_1))
        result.append(self.reve5(x4_0))
        return result
        
class model_nest_split(nn.Module):
    def __init__(self, size1=768, size2=768,num=6):
        super(model_nest_split, self).__init__()
        size_sp=[int(size1/(2**i)) for i in range(num)]
        print('nest split model!!!!!!!!!!!!!!!!!!!!!')
        self.conv1_0 = nn.Sequential(nn.Linear(size_sp[0], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv2_0 = nn.Sequential(nn.Linear(size_sp[1], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv3_0 = nn.Sequential(nn.Linear(size_sp[2], size_sp[3]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv4_0 = nn.Sequential(nn.Linear(size_sp[3], size_sp[4]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv5_0 = nn.Sequential(nn.Linear(size_sp[4], size_sp[5]), ACT2FN["gelu"])#, nn.Dropout(0.1))

        self.conv0_1 = nn.Sequential(nn.Linear(size_sp[0]+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv1_1 = nn.Sequential(nn.Linear(size_sp[1]+size_sp[2], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv2_1 = nn.Sequential(nn.Linear(size_sp[2]+size_sp[3], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv3_1 = nn.Sequential(nn.Linear(size_sp[3]+size_sp[4], size_sp[3]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv4_1 = nn.Sequential(nn.Linear(size_sp[4]+size_sp[5], size_sp[4]), ACT2FN["gelu"])#, nn.Dropout(0.1))

        self.conv0_2 = nn.Sequential(nn.Linear(size_sp[0]*2+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv1_2 = nn.Sequential(nn.Linear(size_sp[1]*2+size_sp[2], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv2_2 = nn.Sequential(nn.Linear(size_sp[2]*2+size_sp[3], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv3_2 = nn.Sequential(nn.Linear(size_sp[3]*2+size_sp[4], size_sp[3]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
        self.conv0_3 = nn.Sequential(nn.Linear(size_sp[0]*3+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv1_3 = nn.Sequential(nn.Linear(size_sp[1]*3+size_sp[2], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv2_3 = nn.Sequential(nn.Linear(size_sp[2]*3+size_sp[3], size_sp[2]), ACT2FN["gelu"])#, nn.Dropout(0.1))

        self.conv0_4 = nn.Sequential(nn.Linear(size_sp[0]*4+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        self.conv1_4 = nn.Sequential(nn.Linear(size_sp[1]*4+size_sp[2], size_sp[1]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
        self.conv0_5 = nn.Sequential(nn.Linear(size_sp[0]*5+size_sp[1], size_sp[0]), ACT2FN["gelu"])#, nn.Dropout(0.1))
        
        self.final=nn.ModuleList()
        for i in range(5):
          self.final.append(nn.Linear(size_sp[0], 2))
          
    def forward(self, x0_0, final=False):
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, x1_0], -1))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0], -1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x1_1], -1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0], -1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1], -1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x1_2], -1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], -1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1], -1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], -1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, x1_3], -1))
        
        x5_0 = self.conv5_0(x4_0)
        x4_1 = self.conv4_1(torch.cat([x4_0, x5_0], -1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, x4_1], -1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, x3_2], -1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, x2_3], -1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x1_4], -1))
        
        result=[x0_5,x0_4,x0_3,x0_2]#,x0_1]
        #result=[x0_1,x0_2,x0_3,x0_4,x0_5]
        if final:
            split_logits=[]
            for i,re in enumerate(result):
                split_logits.append(self.final[i](re))
            return result, split_logits
        else:
            return result
            
class ModelSplitTrainer(nn.Module):
    def __init__(self, size1=768, size2=768,num=6):
        super(ModelSplitTrainer, self).__init__()
        self.model=model_nest_split(size1,size2,num).cuda()
        weight_decay=0.0
        learning_rate=1e-4
        print('model split learning_rate::',learning_rate)
        print('ReLU activation')
        print('12345')
        adam_epsilon=1e-8
        gradient_accumulation_steps=1.0
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        
    def train_step(self, teacher_pos_trans, logits, labels, MINE_split, classifier):
        self.model.train()
        split_results,split_logits=self.model(teacher_pos_trans.detach(), final=True)
        loss_split=0
        for split_feature, split_logit in zip(split_results,split_logits):
        #for split_feature in split_results:
            #split_logit=classifier(split_feature)
            loss_split_cls=torch.nn.CrossEntropyLoss()(split_logit.view(-1,2), labels.view(-1))
            loss_split_dis=MINE_split.compute_MI(split_feature,teacher_pos_trans.detach())
            acc=0.4
            loss_split+=acc * cal_loss(split_logit,logits.detach(), 1) + (1 - acc) * loss_split_cls
            loss_split+=0.1*loss_split_dis
        loss_split=loss_split/len(split_results)
        self.model.zero_grad()
        loss_split.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
             
def get_trans_model(args=None,size1=768,size2=768):
    weight_decay=0.0
    learning_rate=1e-4
    print('model trans learning_rate::',learning_rate)
    print('ReLU activation')
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model = model_trans(size1,size2)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return model.cuda(),optimizer
    
def get_split_model(args=None,size1=768,size2=768):
    weight_decay=0.0
    learning_rate=1e-4
    print('model split learning_rate::',learning_rate)
    print('ReLU activation')
    print('12345')
    adam_epsilon=1e-8
    gradient_accumulation_steps=1.0
    model = model_nest_split(size1,size2)
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
