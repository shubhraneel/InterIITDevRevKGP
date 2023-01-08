import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocess import *

from torch import nn
import torch
import numpy as np
import pandas as pd
import pickle, time
import re, os, string, typing, gc, json
import torch.nn.functional as F
import spacy
from sklearn.model_selection import train_test_split
from collections import Counter
nlp = spacy.load('en_core_web_sm')
# from . import *

# TODO: memory optimization
class SQuAD_Dataset(Dataset):
    def __init__(self, config, df, tokenizer, hide_tqdm=False):
        self.config = config

        self.tokenizer = tokenizer
        self.df = df.reset_index(drop=True)

        # preprocess
        self.data = preprocess_fn(self.df)
        # self.theme_para_id_mapping = self._get_theme_para_id_mapping()

        data_keys = ["answers", "context", "question",
                     "title", "question_id", "context_id", "title_id"]
        
        tokenized_keys = ["question_context_input_ids", "question_context_attention_mask",
                    "start_positions", "end_positions", "answerable",
                    "question_context_offset_mapping"
                    ]

        if not self.config.model.non_pooler:
            tokenized_keys.append("question_context_token_type_ids")

        for key in tokenized_keys:
            self.data[key] = []

        for idx in tqdm(range(0, len(self.data["question"]), self.config.data.tokenizer_batch_size), disable=hide_tqdm):
            example = {key: self.data[key][idx:idx+self.config.data.tokenizer_batch_size] for key in data_keys}

            tokenized_inputs = self._tokenize(example)

            for key in tokenized_keys:
                self.data[key].extend(tokenized_inputs[key])

    # def _get_theme_para_id_mapping(self):
    #     """
    #     Get the indices of all paragraphs of a particular theme  
    #     """
    #     map_ = {}
    #     title_list = list(set(self.data["title"]))
    #     map_ = {title: [i for i in range(len(
    #         self.data["title"])) if title == self.data["title"][i]] for title in title_list}

    #     return map_

    def _tokenize(self, examples):
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
            truncation="only_second" if self.config.data.pad_on_right else "only_first",
            max_length=self.config.data.max_length,
            stride=self.config.data.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True
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
            if (answers["answer_start"] == ""):
                inputs["start_positions"].append(cls_index)
                inputs["end_positions"].append(cls_index)
                inputs["answerable"].append(0)
            else:
                inputs["answerable"].append(1)
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

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

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        inputs["example_id"] = []

        for i in range(len(inputs["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = inputs.sequence_ids(i)
            context_index = 1 if self.config.data.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            inputs["example_id"].append(examples["question_id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            inputs["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(inputs["offset_mapping"][i])
            ]

        inputs["start_positions"] = torch.tensor(inputs["start_positions"])
        inputs["end_positions"] = torch.tensor(inputs["end_positions"])
        inputs["answerable"] = torch.tensor(inputs["answerable"])

        inputs["question_context_input_ids"] = inputs.pop("input_ids")
        inputs["question_context_attention_mask"] = inputs.pop(
            "attention_mask")
        if not self.config.model.non_pooler:
            inputs["question_context_token_type_ids"] = inputs.pop(
                "token_type_ids")
        inputs["question_context_offset_mapping"] = inputs.pop(
            "offset_mapping")

        return inputs

    def __len__(self):
        return len(self.data["question"])

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data.keys()}

    def collate_fn(self, items):
        # batch = {key: torch.stack([x[key] for x in items], dim = 0).squeeze() for key in self.items.keys()}
        # return batch
        batch = {
            "question_context_input_ids":           torch.stack([torch.tensor(x["question_context_input_ids"]) for x in items], dim=0).squeeze(),
            "question_context_attention_mask":      torch.stack([torch.tensor(x["question_context_attention_mask"]) for x in items], dim=0).squeeze(),
            "question_context_offset_mapping":      [x["question_context_offset_mapping"] for x in items],

            "answerable":                           torch.stack([x["answerable"] for x in items], dim=0),
            "start_positions":                      torch.stack([x["start_positions"] for x in items], dim=0),
            "end_positions":                        torch.stack([x["end_positions"] for x in items], dim=0),

            "title":								[x["title"] for x in items],
            "question":								[x["question"] for x in items],
            "context":								[x["context"] for x in items],
            "question_id":							[x["question_id"] for x in items],
            "context_id":							[x["context_id"] for x in items],
            "title_id":								[x["title_id"] for x in items],
            "answer":								[x["answers"]["text"] for x in items],
        }

        if not self.config.model.non_pooler:
            batch["question_context_token_type_ids"] = torch.stack([torch.tensor(
                x["question_context_token_type_ids"]) for x in items], dim=0).squeeze()

        return batch

    def print_row(self, idx, return_dict=False):
        example = self.__getitem__(idx)

        print_dict = {}
        print_dict["answer_text_gold"] = example["answers"]["text"]

        context = example["context"]
        offset_mapping = example["question_context_offset_mapping"]

        start_index = example["start_positions"].item()
        end_index = example["end_positions"].item()

        decoded_answer = ""
        if (start_index != 0 and end_index != 0):
            start_char = offset_mapping[start_index][0]
            end_char = offset_mapping[end_index][1]

            decoded_answer = context[start_char:end_char]

        print_dict["answer_text_decoded"] = decoded_answer

        if (not return_dict):
            print(print_dict)
        else:
            return print_dict





class BiDAF_Dataset(Dataset):
    '''
    - Creates batches dynamically by padding to the length of largest example
      in a given batch.
    - Calulates character vectors for contexts and question.
    - Returns tensors for training.
    '''
    
    def __init__(self, config, df, char2idx, max_context_len = None, max_word_ctx = None, max_question_len = None, max_word_ques = None):
        
        # self.batch_size = batch_size
        # data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        # self.data = data

        self.config = config
        self.df = df
        self.char2idx = char2idx

        self.data = preprocess_fn(self.df)

        print(len(self.data["context"]))
        # self.theme_para_id_mapping = self._get_theme_para_id_mapping()

        data_keys = ["answers", "context", "question",
                     "title", "question_id", "context_id", "title_id"]    

        if max_context_len is None:
            self.max_context_len = max([len(ctx) for ctx in self.data["context"]])
        else:
            self.max_context_len = max_context_len

        if max_word_ctx is None:
            self.max_word_ctx = 0
            for context in tqdm(self.data["context"], position = 0, leave = True, total = len(self.data["context"])):
                for word in nlp(context, disable=['parser','tagger','ner']):
                    if len(word.text) > self.max_word_ctx:
                        self.max_word_ctx = len(word.text)
        else:
            self.max_word_ctx = max_word_ctx

        if max_question_len is None:
            self.max_question_len = max([len(ques) for ques in self.data["question"]])
        else:
            self.max_question_len = max_question_len

        if max_word_ques is None:
            self.max_word_ques = 0
            for question in tqdm(self.data["question"], position = 0, leave = True, total = len(self.data["question"])):
                for word in nlp(question, disable=['parser','tagger','ner']):
                    if len(word.text) > self.max_word_ques:
                        self.max_word_ques = len(word.text)
        else:
            self.max_word_ques = max_word_ques

        self.padded_contexts, self.padded_questions, self.char_contexts, self.char_questions = [], [], [], []
        for idx in tqdm(range(len(self.df)), position = 0, leave = True):

            context_id = self.data["context_ids"][idx]
            context = self.data["context"][idx]
            answer_text = self.data["answers"][idx]["text"]
            
            padded_context = torch.LongTensor(self.max_context_len).fill_(1)
            padded_context[:len(context_id)] = torch.LongTensor(context_id)
            self.padded_contexts.append(padded_context)

            char_context = torch.ones(self.max_context_len, self.max_word_ctx).type(torch.LongTensor)
            char_context = self.make_char_vector(self.max_context_len, self.max_word_ctx, context)
            self.char_contexts.append(char_context)
            
            ques = self.data["question_ids"][idx]
            padded_question = torch.LongTensor(self.max_question_len).fill_(1)
            padded_question[:len(ques)] = torch.LongTensor(ques)
            self.padded_questions.append(padded_question)

            char_ques = torch.ones(self.max_question_len, self.max_word_ques).type(torch.LongTensor)
            char_ques = self.make_char_vector(self.max_question_len, self.max_word_ques, self.data["question"][idx])
            self.char_questions.append(char_ques)

        
    def __len__(self):
        return len(self.data["context"])
    
    def make_char_vector(self, max_sent_len, max_word_len, sentence):
        
        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)
        
        for i, word in enumerate(nlp(sentence, disable=['parser','tagger','ner'])):
            for j, ch in enumerate(word.text):
                if j >= max_word_len:
                    break
                char_vec[i][j] = self.char2idx.get(ch, 0)
        
        return char_vec    
    
    # def get_span(self, text):
        
    #     text = nlp(text, disable=['parser','tagger','ner'])
    #     span = [(w.idx, w.idx+len(w.text)) for w in text]

    #     return span

    def __getitem__(self, idx):
        '''
        Creates batches of data and yields them.
        
        Each yield comprises of:
        :padded_context: padded tensor of contexts for each batch 
        :padded_question: padded tensor of questions for each batch 
        :char_ctx & ques_ctx: character-level ids for context and question
        :label: start and end index wrt context_ids
        :context_text,answer_text: used while validation to calculate metrics
        :ids: question_ids for evaluation
        
        '''

        # context_id = self.data["context_ids"][idx]
        # context = self.data["context"][idx]
        answer_text = self.data["answers"][idx]["text"]
        
        # padded_context = torch.LongTensor(self.max_context_len).fill_(1)
        # padded_context[:len(context_id)] = torch.LongTensor(context_id)
        padded_context = self.padded_contexts[idx]

        # char_context = torch.ones(self.max_context_len, self.max_word_ctx).type(torch.LongTensor)
        # char_context = self.make_char_vector(self.max_context_len, self.max_word_ctx, context)
        char_context = self.char_contexts[idx]
        
        # ques = self.data["question_ids"][idx]
        # padded_question = torch.LongTensor(self.max_question_len).fill_(1)
        # padded_question[:len(ques)] = torch.LongTensor(ques)
        padded_question = self.padded_questions[idx]

        # char_ques = torch.ones(self.max_question_len, self.max_word_ques).type(torch.LongTensor)
        # char_ques = self.make_char_vector(self.max_question_len, self.max_word_ques, self.data["question"][idx])
        char_ques = self.char_questions[idx]

        start_position = self.data["answers"][idx]["answer_start"]
        if start_position == "":
            start_position = 0

        end_position = start_position + len(answer_text)

        return {"padded_context": padded_context,
                "padded_question": padded_question,
                "char_context": char_context,
                "char_question": char_ques,
                "start_positions": torch.tensor(start_position),
                "end_positions": torch.tensor(end_position),

                "context": self.data["context"][idx],
                "answer": answer_text,
                # "id": idx,
                "title": self.data["title"][idx],
                "question": self.data["question"][idx],
                "question_ids": self.data["question_ids"][idx],
                "question_id": self.data["question_id"][idx],
                "context_ids": self.data["context_ids"][idx],
                "context_id": self.data["context_id"][idx],
                "title_id": self.data["title_id"][idx]}


    def collate_fn(self, items):
        # batch = {key: torch.stack([x[key] for x in items], dim = 0).squeeze() for key in self.items.keys()}
        # return batch
        batch = {
            "padded_context":   torch.stack([torch.tensor(x["padded_context"]) for x in items], dim=0).squeeze(),
            "padded_question":  torch.stack([torch.tensor(x["padded_question"]) for x in items], dim=0).squeeze(),
            "char_context":     torch.stack([torch.tensor(x["char_context"]) for x in items], dim=0).squeeze(),
            "char_question":    torch.stack([x["char_question"] for x in items], dim=0),

            "start_positions":  torch.stack([x["start_positions"] for x in items], dim=0),
            "end_positions":    torch.stack([x["end_positions"] for x in items], dim=0),

            "title":	        [x["title"] for x in items],
            "question":	        [x["question"] for x in items],
            "context":	        [x["context"] for x in items],
            "question_id":      [x["question_id"] for x in items],
            "question_ids":      [x["question_ids"] for x in items],
            "context_id":       [x["context_id"] for x in items],
            "context_ids":       [x["context_ids"] for x in items],
            "title_id":	        [x["title_id"] for x in items],
            "answer":	        [x["answer"] for x in items],
        }

        return batch