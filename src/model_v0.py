import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertForSequenceClassification, BertModelForQuestionAnswering

from . import Base_Model

class Bert_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.classifier_model = BertForSequenceClassification.from_pretrained(config.model.model_path, num_labels=config.model.num_labels)

    def training_step(self, batch, batch_idx):
        out = self.classifier_model(batch["question_paragraph_input_ids"], batch["question_paragraph_attention_mask"])

        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim)
        return optimizer


class Bert_QA(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.qa_model = BertModelForQuestionAnswering.from_pretrained(config.model.model_path)

    def training_step(self, batch, batch_idx):
        # TODO: pass answer start and end idx
        out = self.qa_model(batch["question_paragraph_input_ids"], batch["question_paragraph_attention_mask"])

        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim)
        return optimizer


class Bert_Classifier_QA(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """
    def __init__(self, config):
        self.classifier_model = Bert_Classifier(config)
        self.classifier_trainer = pl.Trainer()

        self.qa_model = Bert_QA(config)
        self.qa_model_trainer = pl.Trainer()
        
    def __train__(self, dataloader):
        classifier_trainer.fit(model=self.classifier, train_dataloaders=dataloader)
        qa_model_trainer.fit(model=self.qa_model, train_dataloaders=dataloader)

    def __evaluate__(self, dataloader):
        # TODO
        