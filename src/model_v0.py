import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertForSequenceClassification, BertForQuestionAnswering

from . import Base_Model

class Bert_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.classifier_model = BertForSequenceClassification.from_pretrained(config.model.model_path, num_labels=config.model.num_labels)

    def training_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
                                    attention_mask=batch["question_paragraph_attention_mask"], 
                                    token_type_ids=batch["question_paragraph_token_type_ids"],
                                    labels=batch["answerable"],
                                    )

        print(out)
        
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer


class Bert_QA(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.qa_model = BertForQuestionAnswering.from_pretrained(config.model.model_path)

    def training_step(self, batch, batch_idx):
        # TODO: pass answer start and end idx
        out = self.qa_model(input_ids=batch["question_paragraph_input_ids"], 
                            attention_mask=batch["question_paragraph_attention_mask"],
                            token_type_ids=batch["question_paragraph_token_type_ids"],
                            start_positions=batch["answer_encoded_start_idx"][:, 0],
                            end_positions=batch["answer_encoded_start_idx"][:, 1],
                            )

        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer


class Bert_Classifier_QA(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """
    def __init__(self, config):
        self.classifier_model = Bert_Classifier(config)
        self.classifier_trainer = pl.Trainer(max_epochs=config.training.epochs)

        self.qa_model = Bert_QA(config)
        self.qa_model_trainer = pl.Trainer(max_epochs=config.training.epochs)
        
    def __train__(self, dataloader):
        self.classifier_trainer.fit(model=self.classifier_model, train_dataloaders=dataloader)
        self.qa_model_trainer.fit(model=self.qa_model, train_dataloaders=dataloader)

    def __evaluate__(self, dataloader):
        # TODO
        pass
        