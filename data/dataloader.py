import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data.preprocess import preprocess_fn
from tqdm import tqdm

from torch.utils.data import DataLoader

# TODO: memory optimization
class SQuAD_Dataset(Dataset):
	def __init__(self, config, df, tokenizer, train):
		self.config = config

		self.tokenizer = tokenizer
		self.df = df.reset_index(drop=True)
		self.train = train

		# preprocess
		self.data = preprocess_fn(self.df, self.tokenizer)
		self.theme_para_id_mapping = self._get_theme_para_id_mapping()

		data_keys = ["answers", "context", "id", "question", "title"]

		tokenized_keys = ["question_context_input_ids", "question_context_attention_mask", "question_context_token_type_ids", 
						"title_input_ids", "title_attention_mask", "title_token_type_ids", 
						"context_input_ids", "context_attention_mask", "context_token_type_ids",
						"question_input_ids", "question_attention_mask", "question_token_type_ids",
						"start_positions", "end_positions", "answerable", "question_context_offset_mapping"
						]

		for key in tokenized_keys:
			self.data[key] = []

		# TODO: Parallelise in batches
		for idx in tqdm(range(len(self.data["question"]))):
			example = {key: [self.data[key][idx]] for key in data_keys}

			if self.train:
				tokenized_inputs = self._tokenize_train(example)
			else:
				tokenized_inputs = self._tokenize_test(example)

			for key in tokenized_keys:
				self.data[key].extend(tokenized_inputs[key])

	def _get_theme_para_id_mapping(self):
		"""
		Get the indices of all paragraphs of a particular theme  
		"""
		map_ = {}
		title_list = list(set(self.data["title"]))
		map_ = {title: [i for i in range(len(self.data["title"])) if title == self.data["title"][i]] for title in title_list}
		
		return map_

	def _tokenize_train(self, examples):
		# Some of the questions have lots of whitespace on the left, which is not useful and will make the
		# truncation of the context fail (the tokenized question will take a lots of space). So we remove that
		# left whitespace
		examples["question"] = [q.lstrip() for q in examples["question"]]

		# Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		inputs = self.tokenizer(
			examples["question" if self.config.data.pad_on_right else "context"],
			examples["context" if self.config.data.pad_on_right else "question"],
			truncation = "only_second" if self.config.data.pad_on_right else "only_first",
			max_length = self.config.data.max_length,
			stride = self.config.data.doc_stride,
			return_overflowing_tokens = True,
			return_offsets_mapping = True,
			padding = "max_length",
			# return_tensors = "pt"
		)

		# Since one example might give us several features if it has a long context, we need a map from a feature to
		# its corresponding example. This key gives us just that.
		sample_mapping = inputs.pop("overflow_to_sample_mapping")
		# The offset mappings will give us a map from token to character position in the original context. This will
		# help us compute the start_positions and end_positions.
		offset_mapping = inputs["offset_mapping"]

		# Let's label those examples!
		inputs["start_positions"] = []
		inputs["end_positions"] = []
		inputs["answerable"] = []

		for i, offsets in enumerate(offset_mapping):
			# We will label impossible answers with the index of the CLS token.
			input_ids = inputs["input_ids"][i]
			# cls_index = (input_ids == 2).nonzero(as_tuple = True)[0]
			cls_index = input_ids.index(self.tokenizer.cls_token_id)

			# Grab the sequence corresponding to that example (to know what is the context and what is the question).
			sequence_ids = inputs.sequence_ids(i)

			# One example can give several spans, this is the index of the example containing this span of text.
			sample_index = sample_mapping[i]
			answers = examples["answers"][sample_index]
			# If no answers are given, set the cls_index as answer.
			if len(answers["answer_start"]) == 0:
				inputs["start_positions"].append(cls_index)
				inputs["end_positions"].append(cls_index)
				inputs["answerable"].append(0)
			else:
				inputs["answerable"].append(1)
				# Start/end character index of the answer in the text.
				start_char = answers["answer_start"][0]
				end_char = start_char + len(answers["text"][0])

				# Start token index of the current span in the text.
				token_start_index = 0
				while sequence_ids[token_start_index] != (1 if self.config.data.pad_on_right else 0):
					token_start_index += 1

				# End token index of the current span in the text.
				token_end_index = len(input_ids) - 1
				while sequence_ids[token_end_index] != (1 if self.config.data.pad_on_right else 0):
					token_end_index -= 1

				# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
				if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
					inputs["start_positions"].append(cls_index)
					inputs["end_positions"].append(cls_index)
				else:
					# Otherwise move the token_start_index and token_end_index to the two ends of the answer.
					# Note: we could go after the last offset if the answer is the last word (edge case).
					while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
						token_start_index += 1
					inputs["start_positions"].append(token_start_index - 1)
					while offsets[token_end_index][1] >= end_char:
						token_end_index -= 1
					inputs["end_positions"].append(token_end_index + 1)

		inputs["start_positions"] = torch.tensor(inputs["start_positions"])
		inputs["end_positions"] = torch.tensor(inputs["end_positions"])
		inputs["answerable"] = torch.tensor(inputs["answerable"])

		inputs["question_context_input_ids"] = inputs.pop("input_ids")
		inputs["question_context_attention_mask"] = inputs.pop("attention_mask")
		inputs["question_context_token_type_ids"] = inputs.pop("token_type_ids")
		# inputs["question_context_example_id"] = inputs.pop("example_id")
		inputs["question_context_offset_mapping"] = inputs.pop("offset_mapping")

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

		return inputs

	"""
	TODO:
		Write functions for
			1.	Create an updated data dictionary where each ID contains data corresponding to a data point
			2. 	To retrieve para ids for a particular theme 
					May be create a list of paragraphs corresponding to each theme, if we get a theme, just call that list. (efficient)
					Create this in the init method and store as soon as we get the dataframe 
	"""

	def _tokenize_test(self, examples):
		# Some of the questions have lots of whitespace on the left, which is not useful and will make the
		# truncation of the context fail (the tokenized question will take a lots of space). So we remove that
		# left whitespace
		examples["question"] = [q.lstrip() for q in examples["question"]]

		# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
		# in one example possible giving several features when a context is long, each of those features having a
		# context that overlaps a bit the context of the previous feature.
		inputs = self.tokenizer(
			examples["question" if self.config.data.pad_on_right else "context"],
			examples["context" if self.config.data.pad_on_right else "question"],
			truncation="only_second" if self.config.data.pad_on_right else "only_first",
			max_length=self.config.data.max_length,
			stride=self.config.data.doc_stride,
			return_overflowing_tokens=True,
			return_offsets_mapping=True,
			padding="max_length",
		)

		# Since one example might give us several features if it has a long context, we need a map from a feature to
		# its corresponding example. This key gives us just that.
		sample_mapping = inputs.pop("overflow_to_sample_mapping")

		# The offset mappings will give us a map from token to character position in the original context. This will
		# help us compute the start_positions and end_positions.
		offset_mapping = inputs["offset_mapping"]
		
		# Let's label those examples!
		inputs["start_positions"] = []
		inputs["end_positions"] = []
		inputs["answerable"] = []

		for i, offsets in enumerate(offset_mapping):
			# We will label impossible answers with the index of the CLS token.
			input_ids = inputs["input_ids"][i]
			# cls_index = (input_ids == 2).nonzero(as_tuple = True)[0]
			cls_index = input_ids.index(self.tokenizer.cls_token_id)

			# Grab the sequence corresponding to that example (to know what is the context and what is the question).
			sequence_ids = inputs.sequence_ids(i)

			# One example can give several spans, this is the index of the example containing this span of text.
			sample_index = sample_mapping[i]
			answers = examples["answers"][sample_index]
			# If no answers are given, set the cls_index as answer.
			if len(answers["answer_start"]) == 0:
				inputs["start_positions"].append(cls_index)
				inputs["end_positions"].append(cls_index)
				inputs["answerable"].append(0)
			else:
				inputs["answerable"].append(1)
				# Start/end character index of the answer in the text.
				start_char = answers["answer_start"][0]
				end_char = start_char + len(answers["text"][0])

				# Start token index of the current span in the text.
				token_start_index = 0
				while sequence_ids[token_start_index] != (1 if self.config.data.pad_on_right else 0):
					token_start_index += 1

				# End token index of the current span in the text.
				token_end_index = len(input_ids) - 1
				while sequence_ids[token_end_index] != (1 if self.config.data.pad_on_right else 0):
					token_end_index -= 1

				# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
				if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
					inputs["start_positions"].append(cls_index)
					inputs["end_positions"].append(cls_index)
				else:
					# Otherwise move the token_start_index and token_end_index to the two ends of the answer.
					# Note: we could go after the last offset if the answer is the last word (edge case).
					while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
						token_start_index += 1
					inputs["start_positions"].append(token_start_index - 1)
					while offsets[token_end_index][1] >= end_char:
						token_end_index -= 1
					inputs["end_positions"].append(token_end_index + 1)

		inputs["start_positions"] = torch.tensor(inputs["start_positions"])
		inputs["end_positions"] = torch.tensor(inputs["end_positions"])
		inputs["answerable"] = torch.tensor(inputs["answerable"])

		# We keep the example_id that gave us this feature and we will store the offset mappings.
		inputs["example_id"] = []

		for i in range(len(inputs["input_ids"])):
			# Grab the sequence corresponding to that example (to know what is the context and what is the question).
			sequence_ids = inputs.sequence_ids(i)
			context_index = 1 if self.config.data.pad_on_right else 0

			# One example can give several spans, this is the index of the example containing this span of text.
			sample_index = sample_mapping[i]
			inputs["example_id"].append(examples["id"][sample_index])

			# Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
			# position is part of the context or not.
			inputs["offset_mapping"][i] = [
				(o if sequence_ids[k] == context_index else None)
				for k, o in enumerate(inputs["offset_mapping"][i])
			]

		inputs["question_context_input_ids"] = inputs.pop("input_ids")
		inputs["question_context_attention_mask"] = inputs.pop("attention_mask")
		inputs["question_context_token_type_ids"] = inputs.pop("token_type_ids")
		# inputs["question_context_example_id"] = inputs.pop("example_id")
		inputs["question_context_offset_mapping"] = inputs.pop("offset_mapping")

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

		return inputs

	def __len__(self):
		return len(self.data["question"])

	def __getitem__(self, idx):
		return {key: self.data[key][idx] for key in self.data.keys()}

	def collate_fn(self, items):

		# batch = {key: torch.stack([x[key] for x in items], dim = 0).squeeze() for key in self.items.keys()}
		# return batch

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
			"question_context_input_ids":           torch.stack([torch.tensor(x["question_context_input_ids"]) for x in items], dim=0).squeeze(),
			"question_context_attention_mask":      torch.stack([torch.tensor(x["question_context_attention_mask"]) for x in items], dim=0).squeeze(),
			"question_context_token_type_ids":      torch.stack([torch.tensor(x["question_context_token_type_ids"]) for x in items], dim=0).squeeze(),
			"question_context_offset_mapping":      torch.stack([torch.tensor(x["question_context_offset_mapping"]) for x in items], dim=0).squeeze(),

			"answerable":                           torch.stack([x["answerable"] for x in items], dim=0),
			"start_positions":                      torch.stack([x["start_positions"] for x in items], dim=0),
			"end_positions":                        torch.stack([x["end_positions"] for x in items], dim=0),
			
			"title":								[x["title"] for x in items],
			"question":								[x["question"] for x in items],
			"context":								[x["context"] for x in items],
			"id":									[x["id"] for x in items],
		}

		return batch