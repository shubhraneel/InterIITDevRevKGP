import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data.preprocess import preprocess_fn

# TODO: memory optimization
class SQuAD_Dataset(Dataset):
	def __init__(self, config, df, tokenizer):
		self.config = config

		self.tokenizer = tokenizer

		# preprocess
		self.data = preprocess_fn(df, self.tokenizer)

		# tokenize
		self._tokenize()

	def _tokenize(self):
		# TODO: create a minimalistic version to minimize memory usage
        # TODO: this is inefficient, we should tokenize everything at once
		# TODO: try this - self.data["Theme_tokenized"] 				= self.tokenizer(self.data["Theme"], padding="max_length", truncation="longest_first", return_tensors="pt")
		self.data["Theme_tokenized"] 				= self.data["Theme"].apply(lambda x: self.tokenizer(x, padding="max_length", truncation="longest_first", return_tensors="pt"))
		self.data["Paragraph_tokenized"] 			= self.data["Paragraph"].apply(lambda x: self.tokenizer(x, padding="max_length", truncation="longest_first", return_tensors="pt"))
		self.data["Question_tokenized"] 			= self.data["Question"].apply(lambda x: self.tokenizer(x, padding="max_length", truncation="longest_first", return_tensors="pt"))
		# self.data["Answer_text_tokenized"] 		= self.data["Answer_text"].apply(lambda x: self.tokenizer(x, padding="max_length", truncation="longest_first", return_tensors="pt"))    
		self.data["Question_Paragraph_tokenized"] 	= self.data.apply(lambda x: self.tokenizer(x["Question"], x["Paragraph"], padding="max_length", truncation="only_second", return_tensors="pt"), axis = 1)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		item = {
			"theme":						        self.data.iloc[idx]["Theme_tokenized"],
			"paragraph":						    self.data.iloc[idx]["Paragraph_tokenized"],
			"question":						        self.data.iloc[idx]["Question_tokenized"],
			"question_paragraph":			        self.data.iloc[idx]["Question_Paragraph_tokenized"],

			"answerable":			                torch.from_numpy(np.array(self.data.iloc[idx]["Answer_possible"]).astype(int)),

			# "answer_text":			            self.data.iloc[idx]["Answer_text_tokenized"],
			# "answer_start_idx":			        torch.from_numpy(np.array(self.data.iloc[idx]["Answer_start"])),
			"answer_encoded_start_idx":             torch.from_numpy(np.array(self.data.iloc[idx]["Answer_encoded_start"])),
        }

		return item

	def collate_fn(self, items):
		batch = {
			"theme_input_ids":                      torch.stack([x["theme"]["input_ids"] for x in items], dim=0).squeeze(),
			"theme_attention_mask":                 torch.stack([x["theme"]["attention_mask"] for x in items], dim=0).squeeze(),
			"theme_token_type_ids":                 torch.stack([x["theme"]["token_type_ids"] for x in items], dim=0).squeeze(),
			
			"paragraph_input_ids":                  torch.stack([x["paragraph"]["input_ids"] for x in items], dim=0).squeeze(),
			"paragraph_attention_mask":             torch.stack([x["paragraph"]["attention_mask"] for x in items], dim=0).squeeze(),
			"paragraph_token_type_ids":             torch.stack([x["paragraph"]["token_type_ids"] for x in items], dim=0).squeeze(),

			"question_input_ids":                   torch.stack([x["question"]["input_ids"] for x in items], dim=0).squeeze(),
			"question_attention_mask":              torch.stack([x["question"]["attention_mask"] for x in items], dim=0).squeeze(),
			"question_token_type_ids":              torch.stack([x["question"]["token_type_ids"] for x in items], dim=0).squeeze(),

	        # TODO: eliminate this here, use torch to concatenate q and p in model forward function
			"question_paragraph_input_ids":         torch.stack([x["question_paragraph"]["input_ids"] for x in items], dim=0).squeeze(),
			"question_paragraph_attention_mask":    torch.stack([x["question_paragraph"]["attention_mask"] for x in items], dim=0).squeeze(),
			"question_paragraph_token_type_ids":    torch.stack([x["question_paragraph"]["token_type_ids"] for x in items], dim=0).squeeze(),

			"answerable":                           torch.stack([x["answerable"] for x in items], dim=0),

			# "answer_input_ids":                   torch.stack([x["answer"]["input_ids"] for x in items], dim=0),
			# "answer_attention_mask":              torch.stack([x["answer"]["attention_mask"] for x in items], dim=0),
			# "answer_token_type_ids":              torch.stack([x["answer"]["token_type_ids"] for x in items], dim=0),

			# "answer_start_idx":                   torch.stack([x["answer_start_idx"] for x in items], dim=0),
			"answer_encoded_start_idx":             torch.stack([x["answer_encoded_start_idx"] for x in items], dim=0),
		}

		return batch