import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_f1
import time

def few_shot_calculate_metrics(self, dataloader):

    """
        1. Run the inference script
        2. Calculate the time taken
        3. Calculate the F1 score
        4. Return all
    """

    torch.cuda.synchronize()
    tsince = int(round(time.time() * 1000))
    results = self.__inference__(dataloader)
    torch.cuda.synchronize()
    ttime_elapsed = int(round(time.time() * 1000)) - tsince
    # print ('test time elapsed {}ms'.format(ttime_elapsed))

    ttime_per_example = (ttime_elapsed * dataloader.batch_size)/len(results["ground"])

    f1_spans = []
    for i in range(len(results["predicted_spans"])):
        f1_spans.append(compute_f1(results["predicted_spans"][i], results["gold_spans"][i])) # For the text

    qa_f1 = np.mean(f1_spans)
    return qa_f1, ttime_per_example