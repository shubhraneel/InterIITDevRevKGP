import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
import torch
import numpy as np

class QuestionGeneration(pl.LightningModule):
    def __init__(self, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def forward(self, batch):
        # if "answerable" in batch.keys():
            # out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
            #                         attention_mask=batch["question_paragraph_attention_mask"], 
            #                         token_type_ids=batch["question_paragraph_token_type_ids"],
            #                         labels=batch["answerable"],
            #                         )
        # else:
        out = self.model(
            input_ids=batch['paragraph_input_ids'],
            attention_mask=batch['paragraph_attention_mask'],
            # decoder_input_ids=batch['question_input_ids'],
            # decoder_attention_mask=batch['question_attention_mask'],
            labels=batch['question_input_ids'],
            output_hidden_states=True
        )
    
        return out

    def training_step(self, batch, batch_idx):
        out = self(batch)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        return out.loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer

    def generate(self, test_dataloader):
        self.model.eval()
        outs = []
        for batch in test_dataloader:
            outputs = self.model.generate(input_ids)
            outs += outputs
        return outs


trainer = pl.Trainer()
trainer.train()

