import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data.preprocess import preprocess_fn
from tqdm import tqdm

# TODO: memory optimization
class SQuAD_Dataset(Dataset):
	def __init__(self, config, df, tokenizer, mask_token):
		self.config = config

		self.tokenizer = tokenizer

		# preprocess
		self.data = preprocess_fn(df, self.tokenizer, mask_token)

		# TODO: parallelize in batches

		data_keys = ["answers", "context", "id", "question", "title"]

		tokenized_keys = ["question_context_input_ids", "question_context_attention_mask", "question_context_token_type_ids", 
						"title_input_ids", "title_attention_mask", "title_token_type_ids", 
						"context_input_ids", "context_attention_mask", "context_token_type_ids",
						"question_input_ids", "question_attention_mask", "question_token_type_ids",
						"fewshot_qa_prompt_input_ids", "fewshot_qa_prompt_attention_mask",
						"fewshot_qa_answer_input_ids",
						"start_positions", "end_positions", "answerable"
						]

		fewshot_qa_keys = ["fewshot_qa_prompt", "fewshot_qa_answer"]

		for key in tokenized_keys:
			self.data[key] = []

		# TODO: Parallelise in batches
		for idx in tqdm(range(len(self.data["question"]))):
			example = {key: [self.data[key][idx]] for key in (data_keys + fewshot_qa_keys)}

			tokenized_inputs = self._tokenize(example)

			for key in tokenized_keys:
				self.data[key].extend(tokenized_inputs[key])

	def _tokenize(self, examples):
		questions = [q.strip() for q in examples["question"]]

		inputs = self.tokenizer(
			questions,
			examples["context"],
			max_length=512,
			truncation="only_second",
			return_offsets_mapping=True,
			padding="max_length",
			return_tensors="pt",
		)

		offset_mapping = inputs.pop("offset_mapping")
		answers = examples["answers"]
		start_positions = []
		end_positions = []
		answerable = []

		for i, offset in enumerate(offset_mapping):
			answer = answers[i]

			if (len(answer["answer_start"]) == 0):
				start_positions.append(0)
				end_positions.append(0)
				answerable.append(0)
				continue

			answerable.append(1)

			start_char = answer["answer_start"][0]
			end_char = answer["answer_start"][0] + len(answer["text"][0])
			sequence_ids = inputs.sequence_ids(i)

			# Find the start and end of the context
			idx = 0
			while sequence_ids[idx] != 1:
				idx += 1
			context_start = idx
			while sequence_ids[idx] == 1:
				idx += 1
			context_end = idx - 1

			# If the answer is not fully inside the context, label it (0, 0)
			if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
				start_positions.append(0)
				end_positions.append(0)
			else:
				# Otherwise it's the start and end token positions
				idx = context_start
				while idx <= context_end and offset[idx][0] <= start_char:
					idx += 1
				start_positions.append(idx - 1)

				idx = context_end
				while idx >= context_start and offset[idx][1] >= end_char:
					idx -= 1
				end_positions.append(idx + 1)

		inputs["start_positions"] = torch.tensor(start_positions)
		inputs["end_positions"] = torch.tensor(end_positions)
		inputs["answerable"] = torch.tensor(answerable)

		inputs["question_context_input_ids"] = inputs.pop("input_ids")
		inputs["question_context_attention_mask"] = inputs.pop("attention_mask")
		inputs["question_context_token_type_ids"] = inputs.pop("token_type_ids")

		title_tokenized = self.tokenizer(examples["title"], max_length=512, truncation="longest_first", return_offsets_mapping=True, padding="max_length", return_tensors="pt")
		inputs["title_input_ids"] = title_tokenized["input_ids"]
		inputs["title_attention_mask"] = title_tokenized["attention_mask"]
		inputs["title_token_type_ids"] = title_tokenized["token_type_ids"]

		context_tokenized = self.tokenizer(examples["context"], max_length=512, truncation="longest_first", return_offsets_mapping=True, padding="max_length", return_tensors="pt")    
		inputs["context_input_ids"] = context_tokenized["input_ids"]
		inputs["context_attention_mask"] = context_tokenized["attention_mask"]
		inputs["context_token_type_ids"] = context_tokenized["token_type_ids"]

		question_tokenized = self.tokenizer(examples["question"], max_length=512, truncation="longest_first", return_offsets_mapping=True, padding="max_length", return_tensors="pt")    
		inputs["question_input_ids"] = question_tokenized["input_ids"]
		inputs["question_attention_mask"] = question_tokenized["attention_mask"]
		inputs["question_token_type_ids"] = question_tokenized["token_type_ids"]

		fewshot_qa_prompt_tokenized = self.tokenizer(examples["fewshot_qa_prompt"], max_length=512, truncation="longest_first", padding="max_length", return_tensors="pt")   
		inputs["fewshot_qa_prompt_input_ids"] = fewshot_qa_prompt_tokenized["input_ids"]
		inputs["fewshot_qa_prompt_attention_mask"] = fewshot_qa_prompt_tokenized["attention_mask"]

		fewshot_qa_answer_tokenized = self.tokenizer(examples["fewshot_qa_answer"], max_length=512, truncation="longest_first", padding="max_length", return_tensors="pt")   
		inputs["fewshot_qa_answer_input_ids"] = fewshot_qa_answer_tokenized["input_ids"]

		return inputs

	def __len__(self):
		return len(self.data["question"])

	def __getitem__(self, idx):
		return {key: self.data[key][idx] for key in self.data.keys()}

	def collate_fn(self, items):
		batch = {
			"title_input_ids":                      torch.stack([x["title_input_ids"] for x in items], dim=0).squeeze(),
			"title_attention_mask":                 torch.stack([x["title_attention_mask"] for x in items], dim=0).squeeze(),
			"title_token_type_ids":                 torch.stack([x["title_token_type_ids"] for x in items], dim=0).squeeze(),
			
			"context_input_ids":                    torch.stack([x["context_input_ids"] for x in items], dim=0).squeeze(),
			"context_attention_mask":               torch.stack([x["context_attention_mask"] for x in items], dim=0).squeeze(),
			"context_token_type_ids":               torch.stack([x["context_token_type_ids"] for x in items], dim=0).squeeze(),

			"question_input_ids":                   torch.stack([x["question_input_ids"] for x in items], dim=0).squeeze(),
			"question_attention_mask":              torch.stack([x["question_attention_mask"] for x in items], dim=0).squeeze(),
			"question_token_type_ids":              torch.stack([x["question_token_type_ids"] for x in items], dim=0).squeeze(),

	        # TODO: eliminate this here, use torch to concatenate q and p in model forward function
			"question_context_input_ids":           torch.stack([x["question_context_input_ids"] for x in items], dim=0).squeeze(),
			"question_context_attention_mask":      torch.stack([x["question_context_attention_mask"] for x in items], dim=0).squeeze(),
			"question_context_token_type_ids":      torch.stack([x["question_context_token_type_ids"] for x in items], dim=0).squeeze(),

			"answerable":                           torch.stack([x["answerable"] for x in items], dim=0),
			"start_positions":                      torch.stack([x["start_positions"] for x in items], dim=0),
			"end_positions":                        torch.stack([x["end_positions"] for x in items], dim=0),
			# "answer_input_ids":                   torch.stack([x["answer"]["input_ids"] for x in items], dim=0),
			# "answer_attention_mask":              torch.stack([x["answer"]["attention_mask"] for x in items], dim=0),
			# "answer_token_type_ids":              torch.stack([x["answer"]["token_type_ids"] for x in items], dim=0),

			# "answer_start_idx":                   torch.stack([x["answer_start_idx"] for x in items], dim=0),
			# "answer_encoded_start_idx":           torch.stack([x["answer_encoded_start_idx"] for x in items], dim=0),

			"fewshot_qa_prompt_input_ids":          torch.stack([x["fewshot_qa_prompt_input_ids"] for x in items], dim=0).squeeze(),
			"fewshot_qa_prompt_attention_mask":     torch.stack([x["fewshot_qa_prompt_attention_mask"] for x in items], dim=0).squeeze(),

			"fewshot_qa_answer_input_ids":          torch.stack([x["fewshot_qa_answer_input_ids"] for x in items], dim=0).squeeze(),
		}

		return batch