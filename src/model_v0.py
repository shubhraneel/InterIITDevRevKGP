import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertForSequenceClassification, BertForQuestionAnswering

from . import Base_Model
from tqdm import tqdm

class Bert_Classifier(pl.LightningModule):
    def __init__(self, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()

        self.config = config
        
        self.classifier_model = BertForSequenceClassification.from_pretrained(config.model.model_path, num_labels=config.model.num_labels)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, batch):
        # if "answerable" in batch.keys():
            # out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
            #                         attention_mask=batch["question_paragraph_attention_mask"], 
            #                         token_type_ids=batch["question_paragraph_token_type_ids"],
            #                         labels=batch["answerable"],
            #                         )
        # else:
        out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
                                attention_mask=batch["question_paragraph_attention_mask"], 
                                token_type_ids=batch["question_paragraph_token_type_ids"],
                                )
    
        return out

    def training_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
                                    attention_mask=batch["question_paragraph_attention_mask"], 
                                    token_type_ids=batch["question_paragraph_token_type_ids"],
                                    labels=batch["answerable"],
                                    )
        
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
                                    attention_mask=batch["question_paragraph_attention_mask"], 
                                    token_type_ids=batch["question_paragraph_token_type_ids"],
                                    labels=batch["answerable"],
                                    )

        return out.loss

    def test_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
                                    attention_mask=batch["question_paragraph_attention_mask"], 
                                    token_type_ids=batch["question_paragraph_token_type_ids"],
                                    labels=batch["answerable"],
                                    )

        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer


class Bert_QA(pl.LightningModule):
    def __init__(self, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()

        self.config = config
        
        self.qa_model = BertForQuestionAnswering.from_pretrained(config.model.model_path)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, batch):

        # if "answer_encoded_start_idx" in batch.keys():
        #     out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
        #                         attention_mask = batch["question_paragraph_attention_mask"],
        #                         token_type_ids = batch["question_paragraph_token_type_ids"],
        #                         start_positions = batch["answer_encoded_start_idx"][:, 0],
        #                         end_positions = batch["answer_encoded_start_idx"][:, 1],
        #                         )
        # else:
        out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
                            attention_mask = batch["question_paragraph_attention_mask"],
                            token_type_ids = batch["question_paragraph_token_type_ids"],
                            )

        return out

    def training_step(self, batch, batch_idx):
        # TODO: pass answer start and end idx
        out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
                            attention_mask = batch["question_paragraph_attention_mask"],
                            token_type_ids = batch["question_paragraph_token_type_ids"],
                            start_positions = batch["answer_encoded_start_idx"][:, 0],
                            end_positions = batch["answer_encoded_start_idx"][:, 1],
                            )
        
        return out.loss
    
    def validation_step(self, batch, batch_idx):
        out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
                            attention_mask = batch["question_paragraph_attention_mask"],
                            token_type_ids = batch["question_paragraph_token_type_ids"],
                            )
        return out.loss

    def test_step(self, batch, batch_idx):
        out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
                            attention_mask = batch["question_paragraph_attention_mask"],
                            token_type_ids = batch["question_paragraph_token_type_ids"],
                            )
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config.training.lr)
        return optimizer


class Bert_Classifier_QA(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """
    def __init__(self, config, tokenizer = None):
        self.classifier_model = Bert_Classifier(config)
        self.classifier_trainer = pl.Trainer(max_epochs = config.training.epochs, accelerator = "gpu", devices = 1)

        self.qa_model = Bert_QA(config)
        self.qa_model_trainer = pl.Trainer(max_epochs = config.training.epochs, accelerator = "gpu", devices = 1)

        self.tokenizer = tokenizer
        
    def __train__(self, dataloader):
        self.classifier_trainer.fit(model = self.classifier_model, train_dataloaders = dataloader)
        self.qa_model_trainer.fit(model = self.qa_model, train_dataloaders = dataloader)

    def __inference__(self, dataloader):

        all_preds = []
        all_ground = []
        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            pred = self.classifier_model.predict_step(batch, batch_idx)
            all_preds.extend(torch.argmax(pred.logits, axis = 1).tolist())
            all_ground.extend(batch["answerable"].detach().cpu().numpy())

        all_start_preds = []
        all_end_preds = []
        all_start_ground = []
        all_end_ground = []
        all_input_words = []

        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            pred = self.qa_model.predict_step(batch, batch_idx)
            all_start_preds.extend(torch.argmax(pred.start_logits, axis = 1).tolist())
            all_end_preds.extend(torch.argmax(pred.end_logits, axis = 1).tolist())
            all_start_ground.extend(batch["answer_encoded_start_idx"][:, 0].detach().cpu().numpy())
            all_end_ground.extend(batch["answer_encoded_start_idx"][:, 1].detach().cpu().numpy())
            
            # print(batch["paragraph_input_ids"])
            all_input_words.extend(self.tokenizer.batch_decode(sequences = batch["paragraph_input_ids"]))

        predicted_spans = []
        gold_spans = []

        print(all_start_preds)
        print("\n###########\n")
        print(all_start_ground)

        print("\n###########\n")
        print(all_end_preds)
        print("\n###########\n")
        print(all_end_ground)

        for idx, sentence in enumerate(all_input_words):
            sentence = sentence.split(" ")
            # print(sentence)
            predicted_span = " ".join(sentence[all_start_preds[idx]: all_end_preds[idx]])
            gold_span = " ".join(sentence[all_start_ground[idx]: all_end_ground[idx]])

            predicted_spans.append(predicted_span)
            gold_spans.append(gold_span)

        result = {"preds": all_preds,
                "ground": all_ground,
                "all_start_preds": all_start_preds,
                "all_end_preds": all_end_preds,
                "all_start_ground": all_start_ground,
                "all_end_ground": all_end_ground,
                "all_input_words": all_input_words,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans}
            
        return result

    def __evaluate__(self, dataloader):
        # TODO
        pass