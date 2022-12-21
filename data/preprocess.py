import pandas as pd
from ast import literal_eval

def preprocess_fn(df):
    """
    No preprocessing in v1
    """

    df = df[df['Answer_possible'] == True]

    data_dict = {}
    data_dict["answers"] = []
    data_dict["context"] = []
    data_dict["id"] = []
    data_dict["question"] = []
    data_dict["title"] = []

    # TODO: this is inefficient, we should tokenize everything at once
    for i in range(len(df)):
        answer_start = literal_eval(df.iloc[i]["Answer_start"])
        answer_text = literal_eval(df.iloc[i]["Answer_text"])
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
