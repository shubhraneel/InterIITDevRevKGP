import torch
import numpy as np
import pandas as pd
import spacy
import pickle
from ast import literal_eval
from collections import Counter
import re, os, string, typing, gc, json

from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def preprocess_fn(df, mask_token='<mask>'):
	"""
	No preprocessing in v1
	"""

	data_dict = {}
	data_dict["answers"] = []
	data_dict["context"] = []
	data_dict["question_id"] = []
	data_dict["question_ids"] = []
	data_dict["context_id"] = []
	data_dict["context_ids"] = []
	data_dict["title_id"] = []
	data_dict["question"] = []
	data_dict["title"] = []
	data_dict["fewshot_qa_prompt"] = []
	data_dict["fewshot_qa_answer"] = []

	# question ids 
	ques2idx = {}
	idx2ques = {}

	# df["Answer_start"] = df["Answer_start"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
	# df["Answer_text"] = df["Answer_text"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x) 

	for index, row in df.iterrows():
		# if isinstance(row["Answer_start"], str):
		# 	answer_start = literal_eval(row["Answer_start"])
		# else:
		# 	answer_start = row["Answer_start"]
		
		# if isinstance(row["Answer_text"], str):
		# 	answer_text = literal_eval(row["Answer_text"])
		# else: 
		# 	answer_text = row["Answer_text"]

		answer_start = row["answer_start"]
		answer_text = row["answer_text"]
		
		context = row["context"]
		question_id = row["question_id"]
		
		if "question_ids" in df.columns:
			question_ids = row["question_ids"]
		
		context_id = row["context_id"]
		if "context_ids" in df.columns:
			context_ids = row["context_ids"]

		title_id = row["title_id"]
		question = row["question"]
		title = row["title"]
		answer = {"answer_start": answer_start, "text": answer_text}

		fewshot_qa_prompt = f"Question: {question} Answer: {mask_token} Context: {context}" # 'source_text'
		fewshot_qa_answer = f"Question: {question} Answer: {answer_text}" # 'target_text'
		
		data_dict["answers"].append(answer)
		data_dict["context"].append(context)
		data_dict["question_id"].append(question_id)
		data_dict["context_id"].append(context_id)
		data_dict["title_id"].append(title_id)
		data_dict["question"].append(question)
		data_dict["title"].append(title)
		data_dict["fewshot_qa_prompt"].append(fewshot_qa_prompt)
		data_dict["fewshot_qa_answer"].append(fewshot_qa_answer)

		if "question_ids" in df.columns:
			data_dict["question_ids"].append(question_ids)
		
		if "context_ids" in df.columns:
			data_dict["context_ids"].append(context_ids)

	return data_dict

def load_json(path):
	'''
	Loads the JSON file of the Squad dataset.
	Returns the json object of the dataset.
	'''
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
		
	print("Length of data: ", len(data['data']))
	print("Data Keys: ", data['data'][0].keys())
	print("Title: ", data['data'][0]['title'])
	
	return data




def parse_data(data:dict)->list:
	'''
	Parses the JSON file of Squad dataset by looping through the
	keys and values and returns a list of dictionaries with
	context, query and label triplets being the keys of each dict.
	'''
	data = data['data']
	qa_list = []

	for paragraphs in data:

		for para in paragraphs['paragraphs']:
			context = para['context']

			for qa in para['qas']:
				
				id = qa['id']
				question = qa['question']
				
				for ans in qa['answers']:
					answer = ans['text']
					ans_start = ans['answer_start']
					ans_end = ans_start + len(answer)
					
					qa_dict = {}
					qa_dict['id'] = id
					qa_dict['context'] = context
					qa_dict['question'] = question
					qa_dict['label'] = [ans_start, ans_end]

					qa_dict['answer'] = answer
					qa_list.append(qa_dict)    

	
	return qa_list



def filter_large_examples(df):
	'''
	Returns ids of examples where context lengths, query lengths and answer lengths are
	above a particular threshold. These ids can then be dropped from the dataframe. 
	This is explicitly mentioned in QANet but can be done for other models as well.
	'''
	
	ctx_lens = []
	query_lens = []
	ans_lens = []
	for index, row in df.iterrows():
		ctx_tokens = [w.text for w in nlp(row.context, disable=['parser','ner','tagger'])]
		if len(ctx_tokens)>400:
			ctx_lens.append(row.name)

		query_tokens = [w.text for w in nlp(row.question, disable=['parser','tagger','ner'])]
		if len(query_tokens)>50:
			query_lens.append(row.name)

		ans_tokens = [w.text for w in nlp(row.answer, disable=['parser','tagger','ner'])]
		if len(ans_tokens)>30:
			ans_lens.append(row.name)

		assert row.name == index
	
	return set(ans_lens + ctx_lens + query_lens)


def gather_text_for_vocab(dfs:list):
	'''
	Gathers text from contexts and questions to build a vocabulary.
	
	:param dfs: list of dataframes of SQUAD dataset.
	:returns: list of contexts and questions
	'''
	
	text = []
	total = 0
	for df in dfs:
		unique_contexts = list(df.context.unique())
		unique_questions = list(df.question.unique())
		total += df.context.nunique() + df.question.nunique()
		text.extend(unique_contexts + unique_questions)
	
	assert len(text) == total
	
	return text




def build_word_vocab(vocab_text):
	'''
	Builds a word-level vocabulary from the given text.
	
	:param list vocab_text: list of contexts and questions
	:returns 
		dict word2idx: word to index mapping of words
		dict idx2word: integer to word mapping
		list word_vocab: list of words sorted by frequency
	'''
	
	
	words = []
	for sent in tqdm(vocab_text, position = 0, leave = True, total = len(vocab_text)):
		for word in nlp(sent, disable=['parser','tagger','ner']):
			words.append(word.text)

	word_counter = Counter(words)
	word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
	print(f"raw-vocab: {len(word_vocab)}")
	word_vocab.insert(0, '<unk>')
	word_vocab.insert(1, '<pad>')
	print(f"vocab-length: {len(word_vocab)}")
	word2idx = {word:idx for idx, word in enumerate(word_vocab)}
	print(f"word2idx-length: {len(word2idx)}")
	idx2word = {v:k for k,v in word2idx.items()}
	
	
	return word2idx, idx2word, word_vocab





def build_char_vocab(vocab_text):
	'''
	Builds a character-level vocabulary from the given text.
	
	:param list vocab_text: list of contexts and questions
	:returns 
		dict char2idx: character to index mapping of words
		list char_vocab: list of characters sorted by frequency
	'''
	
	chars = []
	for sent in tqdm(vocab_text, position = 0, leave = True, total = len(vocab_text)):
		for ch in sent:
			chars.append(ch)

	char_counter = Counter(chars)
	char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
	print(f"raw-char-vocab: {len(char_vocab)}")
	high_freq_char = [char for char, count in char_counter.items() if count>=20]
	char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
	print(f"char-vocab-intersect: {len(char_vocab)}")
	char_vocab.insert(0,'<unk>')
	char_vocab.insert(1,'<pad>')
	char2idx = {char:idx for idx, char in enumerate(char_vocab)}
	print(f"char2idx-length: {len(char2idx)}")
	
	return char2idx, char_vocab



def context_to_ids(text, word2idx):
	'''
	Converts context text to their respective ids by mapping each word
	using word2idx. Input text is tokenized using spacy tokenizer first.
	
	:param str text: context text to be converted
	:param dict word2idx: word to id mapping
	:returns list context_ids: list of mapped ids
	
	:raises assertion error: sanity check
	
	'''
	
	context_tokens = [w.text for w in nlp(text, disable=['parser','tagger','ner'])]
	context_ids = [word2idx[word] if word in word2idx.keys() else word2idx["<unk>"] for word in context_tokens]
	
	assert len(context_ids) == len(context_tokens)
	return context_ids



	
def question_to_ids(text, word2idx):
	'''
	Converts question text to their respective ids by mapping each word
	using word2idx. Input text is tokenized using spacy tokenizer first.
	
	:param str text: question text to be converted
	:param dict word2idx: word to id mapping
	:returns list context_ids: list of mapped ids
	
	:raises assertion error: sanity check
	
	'''
	
	question_tokens = [w.text for w in nlp(text, disable=['parser','tagger','ner'])]
	question_ids = [word2idx[word] if word in word2idx.keys() else word2idx["<unk>"] for word in question_tokens]
	
	assert len(question_ids) == len(question_tokens)
	return question_ids
	


	
def test_indices(df, idx2word):
	'''
	Performs the tests mentioned above. This method also gets the start and end of the answers
	with respect to the context_ids for each example.
	
	:param dataframe df: SQUAD df
	:param dict idx2word: inverse mapping of token ids to words
	:returns
		list start_value_error: example idx where the start idx is not found in the start spans
								of the text
		list end_value_error: example idx where the end idx is not found in the end spans
							  of the text
		list assert_error: examples that fail assertion errors. A majority are due to the above errors
		
	'''

	start_value_error = []
	end_value_error = []
	assert_error = []
	for index, row in tqdm(df.iterrows(), position = 0, leave = True, total = len(df)):

		# answer_tokens = [w.text for w in nlp(row['answer_text'], disable=['parser','tagger','ner'])]

		# start_token = row[""][0]
		# end_token = answer_tokens[-1]
		
		context_span  = [(word.idx, word.idx + len(word.text)) 
						 for word in nlp(row['context'], disable=['parser','tagger','ner'])]

		starts, ends = zip(*context_span)

		answer_start = row['answer_start']
		if answer_start == "":
			answer_start = 0
		# print(answer_start, type(answer_start))
		answer_end = answer_start + len(row['answer_text'])

		try:
			start_idx = starts.index(answer_start)
		except:
			start_value_error.append(index)
		try:
			end_idx  = ends.index(answer_end)
		except:
			end_value_error.append(index)

		# try:
		#     assert idx2word[row['context_id'][start_idx]] == answer_tokens[0]
		#     assert idx2word[row['context_id'][end_idx]] == answer_tokens[-1]
		# except:
		#     assert_error.append(index)


	return start_value_error, end_value_error, assert_error



def get_error_indices(df, idx2word):
	
	start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
	err_idx = start_value_error + end_value_error + assert_error
	err_idx = set(err_idx)
	print(f"Number of error indices: {len(err_idx)}")
	
	return err_idx



def index_answer(row, idx2word):
	'''
	Takes in a row of the dataframe or one training example and
	returns a tuple of start and end positions of answer by calculating 
	spans.
	'''
	
	context_span = [(word.idx, word.idx + len(word.text)) for word in nlp(row.context, disable=['parser','tagger','ner'])]
	starts, ends = zip(*context_span)
	
	answer_start = row['answer_start']
	if answer_start == "":
		answer_start = 0
	answer_end = answer_start + len(row['answer_text'])
	start_idx = starts.index(answer_start)
 
	end_idx  = ends.index(answer_end)
	
	ans_toks = [w.text for w in nlp(row.answer_text, disable=['parser','tagger','ner'])]
	ans_start = ans_toks[0]
	ans_end = ans_toks[-1]
	assert idx2word[row.context_id[start_idx]] == ans_start
	assert idx2word[row.context_id[end_idx]] == ans_end
	
	return [start_idx, end_idx]

def preprocess_df(df):
	
	def to_lower(text):
		return text.lower()

	# print(df.columns)

	df.context = df.context.apply(to_lower)
	df.question = df.question.apply(to_lower)
	df.answer_text = df.answer_text.apply(to_lower)

def get_glove_dict():
	'''
	Parses the glove word vectors text file and returns a dictionary with the words as
	keys and their respective pretrained word vectors as values.

	'''
	glove_dict = {}
	with open("./data-dir/glove.6B.100d.txt", "r", encoding="utf-8") as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:], "float32")
			glove_dict[word] = vector
			
	f.close()
	
	return glove_dict

def create_weights_matrix(glove_dict, word_vocab):
	'''
	Creates a weight matrix of the words that are common in the GloVe vocab and
	the dataset's vocab. Initializes OOV words with a zero vector.
	'''
	weights_matrix = np.zeros((len(word_vocab), 100))
	words_found = 0
	for i, word in enumerate(word_vocab):
		try:
			weights_matrix[i] = glove_dict[word]
			words_found += 1
		except:
			pass
		
	return weights_matrix, words_found

