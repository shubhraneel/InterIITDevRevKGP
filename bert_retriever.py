import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class BertRetriever(nn.Module):

    def __init__(self):
        self.bert = BertModel.from_pretrained("bert-base-uncased")
    
    def forward(self, input_ids, context_id, context_vectors):
        x = self.bert(input_ids)
        cls_vec = x[0]
        similarities = cls_vec @ context_vectors.T
        id_onehot = F.one_hot(context_id)
        context_sim = similarities @ id_onehot.T
        total_prob = similarities.exp().sum(axis=1)
        prob = context_sim.exp()/total_prob
        return prob.log().mean()
