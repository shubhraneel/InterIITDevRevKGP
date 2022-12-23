import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
import pandas as pd
from data.preprocess import preprocess_fn
from data.dataloader import SQuAD_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class QuestionAnswerGeneration(pl.LightningModule):
    def __init__(self):

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
        # print(batch['context_input_ids'].size())
        labels = torch.zeros_like(batch["question_answer_input_ids"]) 
        labels[..., :-1] = batch["question_answer_input_ids"][..., 1:]
        labels[..., -1] = 0
        out = self.model(
            input_ids=batch['context_input_ids'],
            attention_mask=batch['context_attention_mask'],
            decoder_input_ids=batch['question_answer_input_ids'],
            decoder_attention_mask=batch['question_answer_attention_mask'],
            labels=labels,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def generate(self, test_dataloader):
        self.model.eval()
        outs = []
        for batch in test_dataloader:
            outputs = self.model.generate(batch['context_input_ids'], attention_mask = batch['context_attention_mask'])
            outs.append(outputs)
        return torch.stack(outs)

tokenizer = T5Tokenizer.from_pretrained('t5-base', TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length")
df = pd.read_csv('data-dir/train_data.csv')
df_train, df_test = train_test_split(df, test_size=0.2, random_state=3407)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=3407)

print(len(df_train))
print(len(df_val))

train_ds = SQuAD_Dataset(df_train, tokenizer)
val_ds = SQuAD_Dataset(df_val, tokenizer)
test_ds = SQuAD_Dataset(df_test, tokenizer)

train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn)
val_dataloader = DataLoader(val_ds, batch_size=4, collate_fn=val_ds.collate_fn)
test_dataloader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn)

print(len(train_dataloader))
print(len(val_dataloader))

model = QuestionAnswerGeneration()

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=4)
trainer.fit(model, train_dataloader, val_dataloader)

generated_out = model.generate(test_dataloader)
decoded_out = [tokenizer.decode(generated_out_item) for generated_out_item in generated_out]
df_test['generated'] = decoded_out
df_test.to_csv("generated_qa.csv")
