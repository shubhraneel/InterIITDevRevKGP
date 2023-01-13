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

    def calculate_metrics(self, dataset, dataloader, logger):

        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        results = self.__inference__(dataset, dataloader, logger)
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time() * 1000)) - tsince


        ttime_per_example = (ttime_elapsed * dataloader.batch_size)/len(results["ground"])
        classification_f1 = sklearn.metrics.f1_score(results["preds"], results["ground"])

        f1_spans = []
        for i in range(len(results["predicted_spans"])):
            f1_spans.append(compute_f1(results["predicted_spans"][i], results["gold_spans"][i]))

        qa_f1 = np.mean(f1_spans)
        return classification_f1, qa_f1, ttime_per_example
