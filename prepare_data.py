import os
import json
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    df = pd.read_csv("data-dir/train_data.csv")
    df["Question"] = df["Theme"] + " " + df["Question"]
    df = df.drop(df.loc[(df.duplicated(subset=["Question"], keep=False)) & (
        df["Answer_possible"] == False)].index, axis=0).reset_index(drop=True)

    ques2idx = {ques: str(idx) for idx, ques in enumerate(df["Question"].unique())}
    idx2ques = {value: key for key, value in ques2idx.items()}

    para2idx = {para: str(idx) for idx, para in enumerate(df["Paragraph"].unique())}
    idx2para = {value: key for key, value in para2idx.items()}

    theme2idx = {theme: str(idx) for idx, theme in enumerate(df["Theme"].unique())}
    idx2theme = {value: key for key, value in theme2idx.items()}

    paragraphs = df['Paragraph']
    para_idx_2_theme_idx = {}
    for idx, row in df.iterrows():
        para_idx_2_theme_idx[para2idx[row['Paragraph']]
                             ] = theme2idx[row['Theme']]

    for i in tqdm(range(len(df))):
        df.loc[i, "question_id"] = ques2idx[df.iloc[i]["Question"]]
        df.loc[i, "paragraph_id"] = para2idx[df.iloc[i]["Paragraph"]]
        df.loc[i, "theme_id"] = theme2idx[df.iloc[i]["Theme"]]

    df.to_csv("data-dir/train_data_prepared.csv", index=False)

    with open("data-dir/para_idx_2_theme_idx.json", "w") as of:
        json.dump(para_idx_2_theme_idx, of)

    with open("data-dir/ques2idx.json", "w") as f:
        json.dump(ques2idx, f)

    with open("data-dir/idx2ques.json", "w") as f:
        json.dump(idx2ques, f)

    with open("data-dir/para2idx.json", "w") as f:
        json.dump(para2idx, f)

    with open("data-dir/idx2para.json", "w") as f:
        json.dump(idx2para, f)

    with open("data-dir/theme2idx.json", "w") as f:
        json.dump(theme2idx, f)

    with open("data-dir/idx2theme.json", "w") as f:
        json.dump(idx2theme, f)
