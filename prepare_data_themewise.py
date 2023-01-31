import os
import pickle
from ast import literal_eval

import pandas as pd
from sklearn.model_selection import *
from tqdm import tqdm

train_frac = 0.6
val_frac = 0.2
test_frac = 0.2
# because 0.1 val and test frac is too small for themewise split
seed = 3407

if __name__ == "__main__":
    df = pd.read_csv("data-dir/train_data.csv").drop("Unnamed: 0", axis=1)
    df = df.rename(
        columns={
            "Question": "question",
            "Theme": "title",
            "Paragraph": "context",
            "Answer_possible": "answerable",
            "Answer_start": "answer_start",
            "Answer_text": "answer_text",
        }
    )

    # concatenate title and question to improve better results and reduce duplicates
    df["question"] = df["title"] + " " + df["question"]

    df = df.drop(
        df.loc[
            (df.duplicated(subset=["question"], keep=False))
            & (df["answerable"] == False)
        ].index,
        axis=0,
    ).reset_index(drop=True)

    ques2idx = {ques: idx for idx, ques in enumerate(df["question"].unique())}
    idx2ques = {value: key for key, value in ques2idx.items()}

    con2idx = {con: idx for idx, con in enumerate(df["context"].unique())}
    idx2con = {value: key for key, value in con2idx.items()}

    title2idx = {title: idx for idx, title in enumerate(df["title"].unique())}
    idx2title = {value: key for key, value in title2idx.items()}

    contexts = df["context"]
    con_idx_2_title_idx = {}
    for idx, row in df.iterrows():
        con_idx_2_title_idx[con2idx[row["context"]]] = title2idx[row["title"]]

    for i in tqdm(range(len(df))):
        df.loc[i, "question_id"] = ques2idx[df.iloc[i]["question"]]
        df.loc[i, "context_id"] = con2idx[df.iloc[i]["context"]]
        df.loc[i, "title_id"] = title2idx[df.iloc[i]["title"]]

    df["answer_start"] = df["answer_start"].apply(lambda x: literal_eval(x))
    df["answer_start"] = df["answer_start"].apply(lambda x: x[0] if len(x) > 0 else "")

    df["answer_text"] = df["answer_text"].apply(lambda x: literal_eval(x))
    df["answer_text"] = df["answer_text"].apply(lambda x: x[0] if len(x) > 0 else "")

    df["question_id"] = df["question_id"].astype(int)
    df["context_id"] = df["context_id"].astype(int)
    df["title_id"] = df["title_id"].astype(int)

    # COMPLETED TODO: Split the dataset in a way where training theme question-context pair should not be split into train/test/val.
    train_idx, val_idx = train_test_split(df, train_size=train_frac, random_state=seed)
    df_train = train_idx.reset_index(drop=True)
    df_val = val_idx.reset_index(drop=True)

    df_val_c = df_val.copy()
    val_idx, test_idx = train_test_split(df_val_c,
        train_size=(val_frac / (test_frac + val_frac)), random_state=seed
    )
    
    df_val = val_idx.reset_index(drop=True)
    df_test = test_idx.reset_index(drop=True)

    print(f"{len(df_train)=}")
    print(f"# themes in df_train {df_train['title'].nunique()}")
    print(f"{len(df_train.loc[df_train['answerable'] == True])=}")

    print(f"{len(df_val)=}")
    print(f"# themes in df_val {df_val['title'].nunique()}")
    print(f"{len(df_val.loc[df_val['answerable'] == True])=}")

    print(f"{len(df_test)=}")
    print(f"# themes in df_test {df_test['title'].nunique()}")
    print(f"{len(df_test.loc[df_test['answerable'] == True])=}")

    os.makedirs("data-dir/train/", exist_ok=True)
    os.makedirs("data-dir/val/", exist_ok=True)
    os.makedirs("data-dir/test/", exist_ok=True)

    df_train.to_pickle("data-dir/train/df_train.pkl")
    df_val.to_pickle("data-dir/val/df_val.pkl")
    df_test.to_pickle("data-dir/test/df_test.pkl")

    print(df)
    del df, df_val_c

    with open("data-dir/con_idx_2_title_idx.pkl", "wb") as f:
        pickle.dump(con_idx_2_title_idx, f)

    with open("data-dir/ques2idx.pkl", "wb") as f:
        pickle.dump(ques2idx, f)

    with open("data-dir/idx2ques.pkl", "wb") as f:
        pickle.dump(idx2ques, f)

    with open("data-dir/con2idx.pkl", "wb") as f:
        pickle.dump(con2idx, f)

    with open("data-dir/idx2con.pkl", "wb") as f:
        pickle.dump(idx2con, f)

    with open("data-dir/title2idx.pkl", "wb") as f:
        pickle.dump(title2idx, f)

    with open("data-dir/idx2title.pkl", "wb") as f:
        pickle.dump(idx2title, f)
