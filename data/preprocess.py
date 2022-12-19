from ast import literal_eval
import pandas as pd

def preprocess_fn(df, sep_token):
    """
    No preprocessing in v1
    """

    # TODO: convert answer_start_idx which is a char_idx to a token_idx (using sentencepiece/byte-pair-encoding directly)

    df["Answer_text"] = df["Answer_text"].apply(lambda x: literal_eval(x))
    df["Answer_start"] = df["Answer_start"].apply(lambda x: literal_eval(x))
	
	df["Question_Paragraph"] = df["Question"] + " " + sep_token + " " + df["Paragraph"]

    return df