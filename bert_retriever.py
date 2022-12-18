import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class BertRetriever(nn.Module):

    def __init__(self, context_path):
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        with open(context_path) as f:
            self.context_dict = pickle.load(f)
    
    def forward(self, x):
        input_ids, attention_mask = x['input_ids'], x['attention_mask']
        context_name = x['context_name']
        context_vectors = self.context_dict[context_name]
        x = self.bert(input_ids, attention_mask)
        cls_vec = x[0]
        similarities = cls_vec @ context_vectors.T
        id_onehot = F.one_hot(context_id)
        context_sim = similarities @ id_onehot.T
        total_prob = similarities.exp().sum(axis=1)
        prob = context_sim.exp()/total_prob
        return -prob.log().mean()
