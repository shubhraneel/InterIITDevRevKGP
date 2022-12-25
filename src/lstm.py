import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from . import Base_Model
from tqdm import tqdm

class LSTM(pl.LightningModule):
    def __init__(self, vocab_size, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()

        self.config = config

        self.hidden_size = config.model.params.hidden_size
        self.num_classes = config.model.params.num_classes
        self.num_layers = config.model.params.num_layers
        self.vocab_size = vocab_size
        self.embedding_size = config.model.params.embedding_size

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(config.model.params.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, config.model.params.num_classes)
        
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def compute_loss(self, output, start_positions, end_positions):
        output = output.view(-1, 2)
        start_logits, end_logits = output.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_positions = start_positions.view(-1)
        end_positions = end_positions.view(-1)

        criterion = nn.CrossEntropyLoss()
        start_loss = criterion(start_logits, start_positions)
        end_loss = criterion(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def forward(self, batch):
        input_ids = batch["question_context_input_ids"].to(torch.long)
        x = self.embedding(input_ids)
       
        sequence_length = input_ids.shape[1]
        out, _ = self.lstm(x)
        out = self.fc(out)

        if "start_positions" in batch.keys():
            # Compute the loss using the start_positions and end_positions here
            start_positions = nn.functional.one_hot(batch["start_positions"], sequence_length).to(torch.float)
            #print(start_positions)
            end_positions = nn.functional.one_hot(batch["end_positions"], sequence_length).to(torch.float)
            #print(end_positions)
            loss = self.compute_loss(out, start_positions, end_positions)
        else:
            loss = None
        
        return out, loss

    def training_step(self, batch, batch_idx):
        _, loss = self.forward(batch)
        self.log('train_loss_qa', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, loss = self.forward(batch)
        self.log('val_loss_qa', loss)
        return loss

    def test_step(self, batch, batch_idx):
        _, loss = self.forward(batch)
        self.log('test_loss_qa', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer

# Define the input and output layers of the model
class LSTMModel(Base_Model):
    def __init__(self, config, tokenizer = None, logger=None):
        self.config = config
        self.logger = logger
        
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        self.qa_model_trainer = pl.Trainer(max_epochs = self.config.training.epochs, accelerator = "gpu", devices = 1, logger=logger)
        self.lstm = LSTM(self.vocab_size, self.config)
  
    def __train__(self, dataloader):
        print("Starting training")

        self.qa_model_trainer.fit(model = self.lstm, train_dataloaders = dataloader)

    def __inference__(self, dataloader):

        all_preds = []
        all_ground = []
        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            all_preds.extend(batch["answerable"].detach().cpu().numpy())
            all_ground.extend(batch["answerable"].detach().cpu().numpy())

        all_start_preds = []
        all_end_preds = []
        all_start_ground = []
        all_end_ground = []
        all_input_words = []

        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            output, _ = self.lstm.predict_step(batch, batch_idx)
            #output = output.view(-1, 2)
            start_logits, end_logits = output.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            all_start_preds.extend(torch.argmax(start_logits, axis=1).tolist())
            all_end_preds.extend(torch.argmax(end_logits, axis=1).tolist())
            all_start_ground.extend(batch["start_positions"].detach().cpu().numpy())
            all_end_ground.extend(batch["end_positions"].detach().cpu().numpy())
            
            all_input_words.extend(self.tokenizer.batch_decode(sequences = batch["context_input_ids"]))

        predicted_spans = []
        gold_spans = []

        for idx, sentence in enumerate(all_input_words):
            sentence = sentence.split(" ")
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