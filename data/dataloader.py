import pandas as pd

import torch
from import torch.utils.data Dataset

from . import preprocess_fn

class SQuAD_Dataset(Dataset):
	def __init__(self, config, df, tokenizer):
		self.config = config

		self.tokenizer = tokenizer

		# preprocess
		self.data = preprocess_fn(df)

		# tokenize
		self._tokenize()

	def _tokenize(self):
		# TODO: create a minimalistic version to minimize memory usage

		self.data["Theme_tokenized"] 		= self.data["Theme"].apply(lambda x: self.tokenizer(x))
		self.data["Paragraph_tokenized"] 	= self.data["Paragraph"].apply(lambda x: self.tokenizer(x))
		self.data["Question_tokenized"] 	= self.data["Question"].apply(lambda x: self.tokenizer(x))
		self.data["Answer_text_tokenized"] 	= self.data["Answer_text"].apply(lambda x: self.tokenizer(x))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		theme = self.data.iloc[idx]["Theme_tokenized"]
		paragraph = self.data.iloc[idx]["Paragraph_tokenized"]
		question = self.data.iloc[idx]["Question_tokenized"]

		answerable = self.data.iloc[idx]["Answer_possible"]
	
		answer_text = self.data.iloc[idx]["Answer_text_tokenized"]
		answer_start_idx = self.data.iloc[idx]["Answer_start"]

		return theme, paragraph, question, answerable, answer_text, answer_start_idx

	def collate_fn(self, items):
		batch = {
			"theme_input_ids": torch.stack([x[0]["input_ids"] for x in items], dim=0),
			"theme_attention_mask": torch.stack([x[0]["attention_mask"] for x in items], dim=0),
			
			"paragraph_input_ids": torch.stack([x[1]["input_ids"] for x in items], dim=0),
			"paragraph_attention_mask": torch.stack([x[1]["attention_mask"] for x in items], dim=0),

			"question_input_ids": torch.stack([x[2]["input_ids"] for x in items], dim=0),
			"question_attention_mask": torch.stack([x[2]["attention_mask"] for x in items], dim=0),

			"answerable": torch.stack([x[3] for x in items], dim=0),

			"answer_input_ids": torch.stack([x[4]["input_ids"] for x in items], dim=0),
			"answer_attention_mask": torch.stack([x[4]["attention_mask"] for x in items], dim=0),

			"answer_start_idx": torch.stack([x[5] for x in items], dim=0),
		}

		return batch