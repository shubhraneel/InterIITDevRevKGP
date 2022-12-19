import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import sklearn

from . import compute_f1

class Base_Model():
    def __init__(self):
        pass

    def __train__(self):
        raise NotImplementedError("No training method implemented")

    def __evaluate__(self):
        raise NotImplementedError("No evaluation method implemented")

    def calculate_metrics(self):
        # TODO: add logic here
        torch.cuda.synchronize()
        tsince = int(round(time.time()*1000))
        Pred_tf, Gold_tf, Pred_span, Gold_span = self.__evaluate__()

        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time()*1000)) - tsince
        print ('test time elapsed {}ms'.format(ttime_elapsed))

        ttime_per_example = ttime_elapsed/len(Pred_tf)
        F1_tf = sklearn.metrics.f1_score(Pred_tf, Gold_tf)#For paragraph search task
        F1_span = compute_f1(Pred_span, Gold_span)#For the text
        wandb.log({"F1_tf": F1_tf},{"F1_span": F1_span},{"ttime_per_example": ttime_per_example})#Log into wandb

        return F1_tf, F1_span, ttime_per_example