from ast import literal_eval
import pandas as pd

def preprocess_fn(df, tokenizer):
    """
    No preprocessing in v1
    """

    # TODO: convert answer_start_idx which is a char_idx to a token_idx (using sentencepiece/byte-pair-encoding directly)

    df["Answer_text"] = df["Answer_text"].apply(lambda x: literal_eval(x))
    df["Answer_start"] = df["Answer_start"].apply(lambda x: literal_eval(x))
    df["Answer_encoded_start"]  = df.apply(lambda x: find_position(x["Paragraph"], x["Answer_text"], x["Answer_start"], tokenizer))

    return df

def find_position(Paragraph, Answer_text, Answer_start, tokenizer):
    encoded_s = tokenizer(Paragraph)
    encoded_s_sliced = tokenizer(Paragraph[id:])
    encoded_t = tokenizer(Answer_text)
    tokenizer.decode(encoded_t['input_ids'])
    len_t = len(encoded_t['input_ids']) - 2
    start = len(encoded_s["input_ids"]) - len(encoded_t["input_ids"])
    end = start + len_t

    return start, end
