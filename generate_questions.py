import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
import pandas as pd
from data.preprocess import preprocess_fn
from data.dataloader import SQuAD_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ast import literal_eval

class QuestionGeneration(pl.LightningModule):
    def __init__(self):

        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

    def forward(self, batch):
        # if "answerable" in batch.keys():
            # out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
            #                         attention_mask=batch["question_paragraph_attention_mask"], 
            #                         token_type_ids=batch["question_paragraph_token_type_ids"],
            #                         labels=batch["answerable"],
            #                         )
        # else:
        # print(batch['context_input_ids'].size())
        # labels = torch.zeros_like(batch["question_answer_input_ids"]) 
        # labels[..., :-1] = batch["question_answer_input_ids"][..., 1:]
        # labels[..., -1] = 0
        out = self.model(
            input_ids=batch['answer_context_input_ids'],
            attention_mask=batch['answer_context_attention_mask'],
            # decoder_input_ids=batch['question_answer_input_ids'],
            # decoder_attention_mask=batch['question_answer_attention_mask'],
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def generate(self, test_dataloader):
        self.model.eval()
        outs = []
        for batch in test_dataloader:
            outputs = self.model.generate(
                batch['answer_context_input_ids'], 
                attention_mask = batch['answer_context_attention_mask'],
                num_beams=5,
                num_return_sequences=5
            )
            print(outputs.shape)
            outs.append(outputs)
        return torch.stack(outs)
        
tokenizer = T5Tokenizer.from_pretrained('mrm8488/t5-base-finetuned-question-generation-ap', TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length")
df = pd.read_csv('data-dir/train_data.csv')
df = preprocess_fn(df)
df.pop("id")
df.pop("question")
df_generated = pd.read_csv('/content/drive/MyDrive/SyntheticGeneration/AGeneration/generated_a.csv')
df_generated['generated_answers'] = literal_eval(df_generated['generated_answers'])
df_generated['generated_answer_indices'] = literal_eval(df_generated['generated_answer_indices'])
for idx, row in df_generated.iterrows():
    for idx_ans, answer in enumerate(row['generated_answers']):
        df['answers'].append({
          "answer_start": row['generated_answers'][idx_ans],
          "text": answer
        })
        df["context"].append(row["context"])
        df["title"].append(row["title"])

ds = SQuAD_Dataset(df, tokenizer)

dataloader = DataLoader(ds, batch_size=4, collate_fn=test_ds.collate_fn)

print(len(train_dataloader))
print(len(val_dataloader))

model = QuestionGeneration.load_from_checkpoint("/content/drive/SyntheticGeneration/QGeneration/**")

generated_out = model.generate(dataloader)
decoded_out = [tokenizer.decode(generated_out_item) for generated_out_item in generated_out]
df_test['generated'] = decoded_out
df_test.to_csv("/content/drive/SyntheticGeneration/QGeneration/generated_q.csv")
