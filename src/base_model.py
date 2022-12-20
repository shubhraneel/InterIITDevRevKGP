import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_f1
import time
import sklearn



class Base_Model():
    def __init__(self):
        pass

    def __train__(self):
        raise NotImplementedError("No training method implemented")

    def __evaluate__(self):
        raise NotImplementedError("No evaluation method implemented")

    def __inference__(self):
        raise NotImplementedError("No inference method implemented")

    def calculate_metrics(self, dataloader):

        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        # TODO: add logic here 
        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        results = self.__inference__(dataloader)
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time()*1000)) - tsince
        print ('test time elapsed {}ms'.format(ttime_elapsed))

        ttime_per_example = ttime_elapsed/len(results["ground"])
        F1_tf = sklearn.metrics.f1_score(results["preds"], results["ground"]) # For paragraph search task

        # startidx endidx
        # A B C D R G Y H H H P W L --- start = 2, end = 5
        # B C D R G Y H H H P W L A --- start = 3, end = 7

        # print(results["all_input_words"])
        # exit(0)

        f1_spans = []
        for i in range(len(results["predicted_spans"])):
            f1_spans.append(compute_f1(results["predicted_spans"], results["gold_spans"])) # For the text
        # wandb.log({"F1_tf": F1_tf},{"F1_span": F1_span},{"ttime_per_example": ttime_per_example})#Log into wandb

        F1_span = np.mean(f1_spans)
        return F1_tf, F1_span, ttime_per_example