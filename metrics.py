import torch
import pytorch_lightning as pl
import re
import string
import sys
from collections import Counter
import wandb
import sklearn



def CalcMetrics(model):
  torch.cuda.synchronize()
  tsince = int(round(time.time()*1000))
  Pred_tf, Gold_tf, Pred_span, Gold_span = model
  torch.cuda.synchronize()
  ttime_elapsed = int(round(time.time()*1000)) - tsince
  print ('test time elapsed {}ms'.format(ttime_elapsed))
  
  ttime_per_example = ttime_elapsed/len(Pred_tf)
  F1_tf = sklearn.metrics.f1_score(Pred_tf, Gold_tf)#For paragraph search task
  F1_span = f1_score(Pred_span, Gold_span)#For the text
  wandb.log({"F1_tf": F1_tf},{"F1_span": F1_span},{"ttime_per_example": ttime_per_example})#Log into wandb
  
  return F1_tf, F1_span, ttime_per_example

def f1_score(prediction, ground_truth):
  precision = 0
  recall = 0
  for i in range(len(prediction)):
    prediction_tokens = normalize_answer(prediction[i]).split()#text preprocessing
    ground_truth_tokens = normalize_answer(ground_truth[i]).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    precision = precision + 1.0 * num_same / len(prediction_tokens)
    recall = recall + 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))
