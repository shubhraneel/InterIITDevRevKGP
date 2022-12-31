import pandas as pd
from ast import literal_eval

def preprocess_fn(df, tokenizer, mask_token='<mask>'):
	"""
	No preprocessing in v1
	"""

	# TODO: Fix context and paragraph everywhere (also title, theme)
	data_dict = {}
	data_dict["answers"] = []
	data_dict["context"] = []
	data_dict["question_id"] = []
	data_dict["paragraph_id"] = []
	data_dict["theme_id"] = []
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
		if isinstance(row["Answer_start"], str):
			answer_start = literal_eval(row["Answer_start"])
		else:
			answer_start = row["Answer_start"]
		
		if isinstance(row["Answer_text"], str):
			answer_text = literal_eval(row["Answer_text"])
		else: 
			answer_text = row["Answer_text"]
		
		context = row["Paragraph"]
		question_id = row["question_id"]
		paragraph_id = row["paragraph_id"]
		theme_id = row["theme_id"]
		question = row["Question"]
		title = row["Theme"]
		answer = {"answer_start": answer_start, "text": answer_text}

		fewshot_qa_prompt = f"Question: {question} Answer: {mask_token} Context: {context}" # 'source_text'
		fewshot_qa_answer = f"Question: {question} Answer: {answer_text}" # 'target_text'
		
		data_dict["answers"].append(answer)
		data_dict["context"].append(context)
		data_dict["question_id"].append(question_id)
		data_dict["paragraph_id"].append(paragraph_id)
		data_dict["theme_id"].append(theme_id)
		data_dict["question"].append(question)
		data_dict["title"].append(title)
		data_dict["fewshot_qa_prompt"].append(fewshot_qa_prompt)
		data_dict["fewshot_qa_answer"].append(fewshot_qa_answer)

	return data_dict
