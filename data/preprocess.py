import pandas as pd
from ast import literal_eval

def preprocess_fn(df, tokenizer, mask_token):
	"""
	No preprocessing in v1
	"""

	data_dict = {}
	data_dict["answers"] = []
	data_dict["context"] = []
	data_dict["id"] = []
	data_dict["question"] = []
	data_dict["title"] = []
	data_dict["fewshot_qa_prompt"] = []
	data_dict["fewshot_qa_answer"] = []

	for index, row in df.iterrows():
		answer_start = literal_eval(row["Answer_start"])
		answer_text = literal_eval(row["Answer_text"])
		
		context = row["Paragraph"]
		id = row["Unnamed: 0"]
		question = row["Question"]
		title = row["Theme"]
		answer = {"answer_start": answer_start, "text": answer_text}

		fewshot_qa_prompt = f"Question: {question} Answer: {mask_token} Context: {context}" # 'source_text'
		fewshot_qa_answer = f"Question: {question} Answer: {answer_text}" # 'target_text'
		
		data_dict["answers"].append(answer)
		data_dict["context"].append(context)
		data_dict["id"].append(id)
		data_dict["question"].append(question)
		data_dict["title"].append(title)
		data_dict["fewshot_qa_prompt"].append(fewshot_qa_prompt)
		data_dict["fewshot_qa_answer"].append(fewshot_qa_answer)

	return data_dict
