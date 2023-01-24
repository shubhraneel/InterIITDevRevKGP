import concurrent.futures
import re
import string
import time

import matplotlib.pyplot as plt

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("wordnet")

sw = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def collate_paragraphs(df):
    collated_df = df.groupby("Theme", as_index=False).aggregate(lambda x: x.tolist())
    return collated_df


def co_app(para, ques):
    return len(set(para.split()) & set(ques.split()))


def co_appearance_on_collated_df(collated_df):
    co_appearance = []
    for idx, row in collated_df.iterrows():
        topic_co_appearance = []
        for para, ques in zip(row["Paragraph"], row["Question"]):
            topic_co_appearance.append(len(set(para.split()) & set(ques.split())))
        co_appearance.append(topic_co_appearance)
    collated_df["co_appearance_scores"] = co_appearance
    return collated_df


def preprocess(text):
    text = text.casefold()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(r"http\S+", "", text)
    html = re.compile(r"<.*?>")
    text = html.sub(r"", text)
    punctuations = "@#!?+&*[]-%.:/();$=><|{}^" + "'`" + "_"
    for p in punctuations:
        text = text.replace(p, "")  # Removing punctuations
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)  # removing stopwords
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)  # Removing emojis
    return text


def to_target(text):
    if text == True:
        return 1
    return 0


if __name__ == "__main__":
    df = pd.read_csv("data-dir/train_data.csv")
    # collated_df=collate_paragraphs(df)
    # collated_df=co_appearance_on_collated_df(collated_df)
    # collated_df.to_csv("data-dir/collated.csv")
    with concurrent.futures.ProcessPoolExecutor(8) as pool:
        df["Paragraph"] = list(
            tqdm(pool.map(preprocess, df["Paragraph"], chunksize=5000))
        )
        df["Question"] = list(
            tqdm(pool.map(preprocess, df["Question"], chunksize=5000))
        )
        df["target"] = list(
            tqdm(pool.map(to_target, df["Answer_possible"], chunksize=5000))
        )
        df["coappearance"] = list(
            tqdm(pool.map(co_app, df["Question"], df["Paragraph"], chunksize=5000))
        )

    df.to_csv("data-dir/preprocessed.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        df["target"].values,
        test_size=0.2,
        random_state=123,
        stratify=df["target"].values,
    )

    ### TO:DO
    ### try with doc2vec, word2vec avg, infersent avg

    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_vectorizer.fit(X_train["Paragraph"])
    tfidf_p_train = tfidf_vectorizer.transform(X_train["Paragraph"])
    tfidf_q_train = tfidf_vectorizer.transform(X_train["Question"])
    print(len(tfidf_vectorizer.vocabulary_))
    print(tfidf_p_train.shape)
    print(tfidf_q_train.shape)

    cosine_similarity_train = np.vstack(
        [
            cosine_similarity(tfidf_p_train[i], tfidf_q_train[i])
            for i in range(tfidf_p_train.shape[0])
        ]
    )
    # print(cosine_similarity_train.shape)

    X_train = pd.DataFrame(X_train["coappearance"])
    X_train["cosine"] = cosine_similarity_train
    print(X_train.head())

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    tsince = int(round(time.time() * 1000))
    tfidf_p_test = tfidf_vectorizer.transform(X_test["Paragraph"])
    tfidf_q_test = tfidf_vectorizer.transform(X_test["Question"])
    # print(tfidf_p_test.shape)
    # print(tfidf_q_test.shape)
    cosine_similarity_test = np.vstack(
        [
            cosine_similarity(tfidf_p_test[i], tfidf_q_test[i])
            for i in range(tfidf_p_test.shape[0])
        ]
    )
    # print(cosine_similarity_test.shape)

    X_test = pd.DataFrame(X_test["coappearance"])
    X_test["cosine"] = cosine_similarity_test
    # print(X_test.head())
    y_pred = classifier.predict(X_test)
    ttime_elapsed = int(round(time.time() * 1000)) - tsince
    ttime_per_example = ttime_elapsed / X_test.shape[0]
    print(f"test time elapsed {ttime_elapsed} ms")
    print(f"test time elapsed per example {ttime_per_example} ms")
    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cnf_matrix, annot=labels, fmt="", cmap="Blues")
    plt.show()
