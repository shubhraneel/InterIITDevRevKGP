import pandas as pd
from ast import literal_eval
# from datasets import load_dataset, Dataset


def preprocess_fn(df, tokenizer):
    """
    No preprocessing in v1
    """

    # df["Answer_text"] = df["Answer_text"].apply(lambda x: literal_eval(x))
    # df["Answer_start"] = df["Answer_start"].apply(lambda x: literal_eval(x))
    
    data_dict = {}
    data_dict["answers"] = []
    data_dict["context"] = []
    data_dict["id"] = []
    data_dict["question"] = []
    data_dict["title"] = []

    # TODO: this is inefficient, we should tokenize everything at once
    # df["Answer_encoded_start"]  = df.apply(lambda x: find_position(x.Paragraph, x.Answer_text, x.Answer_start, tokenizer), axis = 1)
    # df["Answer_start"] = df["Answer_start"].apply(lambda x: x if x != [] else [0])

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

    # dataset = Dataset.from_dict(data_dict)
    return data_dict

# def find_position(Paragraph, Answer_text, Answer_start, tokenizer):
# 	# TODO: optimize this so that tokenizers are called only once
#     encoded_s = tokenizer(Paragraph)
#     if len(Answer_start) != 0:
#         Answer_start = Answer_start[0]
#     else:
#         return 0, 0
#     encoded_s_sliced = tokenizer(Paragraph[Answer_start:])
#     encoded_t = tokenizer(Answer_text)
#     len_t = len(encoded_t['input_ids']) - 2
#     start = len(encoded_s["input_ids"]) - len(encoded_t["input_ids"])
#     end = start + len_t

#     return start, end


# """
#     paragraph - Elon Musk is a very bla bla bla bla 
#     answer = very bla
#     answer_start = 15
#     answer_start = 4
#     answer_end = 5
# """