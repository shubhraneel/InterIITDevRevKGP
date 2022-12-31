import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
import pandas as pd
from data.preprocess import preprocess_fn
from data.dataloader import AllAnswers_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda:0')

class AnswerGeneration(pl.LightningModule):
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
        # labels = torch.zeros_like(batch["question_answer_input_ids"]) 
        # labels[..., :-1] = batch["question_answer_input_ids"][..., 1:]
        # labels[..., -1] = 0
        out = self.model(
            input_ids=batch['context_input_ids'],
            attention_mask=batch['context_attention_mask'],
            # decoder_input_ids=batch['question_answer_input_ids'],
            # decoder_attention_mask=batch['question_answer_attention_mask'],
            labels=batch['answer_input_ids'],
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
        self.model.to(device)
        outs = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids, attn_mask = batch['context_input_ids'], batch['context_attention_mask']
                outputs = self.model.generate(input_ids.to(device), attention_mask = attn_mask.to(device), max_new_tokens=50)
                outs.extend(list(outputs))
        return outs

tokenizer = T5Tokenizer.from_pretrained('t5-base', TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length")
df = pd.read_json('data-dir/train_data.json', orient='records')
ds = AllAnswers_Dataset(df, tokenizer)
dataloader = DataLoader(ds, batch_size=8, collate_fn=ds.collate_fn)
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=3407)
# df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=3407)

# print(len(df_train))
# print(len(df_val))

# train_ds = AllAnswers_Dataset(df_train, tokenizer)
# val_ds = AllAnswers_Dataset(df_val, tokenizer)
# test_ds = AllAnswers_Dataset(df_test, tokenizer)

# train_dataloader = DataLoader(train_ds, batch_size=4, collate_fn=train_ds.collate_fn)
# val_dataloader = DataLoader(val_ds, batch_size=4, collate_fn=val_ds.collate_fn)
# test_dataloader = DataLoader(test_ds, batch_size=4, collate_fn=test_ds.collate_fn)

# print(len(train_dataloader))
# print(len(val_dataloader))

# model = AnswerGeneration()

# trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=4, default_root_dir="/content/drive/MyDrive/SyntheticGeneration/AGeneration/models/")
# trainer.fit(model, train_dataloader, val_dataloader)

model = AnswerGeneration.load_from_checkpoint("/content/drive/MyDrive/SyntheticGeneration/AGeneration/models/lightning_logs/version_1/checkpoints/epoch=3-step=11200.ckpt")

generated_out = model.generate(dataloader)
decoded_out = [tokenizer.decode(generated_out_item) for generated_out_item in generated_out]

def process_out(context, x):
    context = context.lower()
    x = x[6:]
    x = x.split("</s>")[0]
    x = x.split("<extra_id_0>")
    x = [x_i.strip().lower() for x_i in x]
    x = list(set(x))
    x = [x_i for x_i in x if len(x_i) > 0]
    x = [x_i for x_i in x if x_i in context]
    return x

def find_char_index(context, answers):
    context = context.lower()
    answer_indices = [context.index(answer) for answer in answers]
    return answer_indices

df['generated_answers_raw'] = decoded_out
df['generated_answers'] = df.apply(lambda x: process_out(x['context'], x['generated_answers_raw']), axis=1)
df['generated_answer_indices'] = df.apply(lambda x: find_char_index(x['context'], x['generated_answers']), axis=1)
df.to_csv("/content/drive/MyDrive/SyntheticGeneration/AGeneration/generated_a.csv")
