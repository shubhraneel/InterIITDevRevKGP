import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModelForQuestionAnswering

class BertQA(nn.Module):

    def __init__(self):
        self.bert = BertModelForQuestionAnswering.from_pretrained("bert-base-cased")
    
    def forward(self, x):
        x = self.bert(**x)
        return x[:2]
