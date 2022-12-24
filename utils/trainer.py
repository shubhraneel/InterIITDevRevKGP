import numpy as np
import pandas as pd
from tqdm import tqdm


class Trainer():

    def __init__(self, config, model, optimizer, device):

        self.config = config
        self.model = model
        self.device = device

        self.optimizer = optimizer
    
    def _train_step(self, dataloader, epoch):
        
        total_loss = 0
        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for batch_idx, batch in tqdm(enumerate(tepoch), position = 0, leave = True):
                tepoch.set_description(f"Epoch {epoch + 1}")

                out = self.model(batch)
                loss = out.loss
                loss.backward()

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))
                
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / batch_idx

                
    def train(self, dataloader):

        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(dataloader, epoch)
            
            if (epoch + 1) % self.config.training.evaluate_every:
                self.evaluate(dataloader)


    def _evaluate_step(self, dataloader, epoch = None):
        
        total_loss = 0
        with tqdm(dataloader, unit = "batch", position = 0, leave = True) as tepoch:
            for batch_idx, batch in tqdm(enumerate(tepoch), position = 0, leave = True):
                tepoch.set_description(f"Epoch {epoch + 1}")

                out = self.model(batch)
                loss = out.loss

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))

        return total_loss / batch_idx
    
    def evaluate(self, dataloader):
        self._evaluate_step(dataloader)

    # Add internal functions for inference here

    def inference(self):
        pass
        

