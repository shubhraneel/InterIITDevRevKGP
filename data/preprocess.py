import pandas as pd
from ast import literal_eval

def preprocess_fn(df, tokenizer):
    """
    No preprocessing in v1
    """

    data_dict = {}
    data_dict["answers"] = []
    data_dict["context"] = []
    data_dict["id"] = []
    data_dict["question"] = []
    data_dict["title"] = []

    # TODO: this is inefficient, we should tokenize everything at once
    for i in range(len(df)):

        if isinstance(df.iloc[i]["Answer_start"], str):
            answer_start = literal_eval(df.iloc[i]["Answer_start"])
        else:
            answer_start = df.iloc[i]["Answer_start"]
        
        if isinstance(df.iloc[i]["Answer_text"], str):
            answer_text = literal_eval(df.iloc[i]["Answer_text"])
        else: 
            answer_text = df.iloc[i]["Answer_text"]

        context = df.iloc[i]["Paragraph"]
        id = df.iloc[i]["Unnamed: 0"]
        question = df.iloc[i]["Question"]
        title = df.iloc[i]["Theme"]

        answer = {"answer_start": answer_start, "text": answer_text}
        data_dict["answers"].append(answer)

        data_dict["context"].append(context)
        data_dict["id"].append(id)
        data_dict["question"].append(question)
        data_dict["title"].append(title)

    return data_dict
