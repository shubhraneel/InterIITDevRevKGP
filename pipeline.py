def set_seed(seed=0, verbose=False):
    if seed is None:
        seed = int(show_time())
    if verbose: print("[Info] seed set to: {}".format(seed))

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def read_data(): 
    df=pd.read_csv("train_data.csv")
    df=df.drop(labels=["Unnamed: 0"],axis="columns")
    return df

def split(df, train_size, val_size, test_size, rando):
    # train_val_test split
    themes=list(df["Theme"].unique())
    train_themes,test_themes=train_test_split(themes,train_size=(train_size+val_size)/100, random_state=rando)
    train_themes,val_themes=train_test_split(themes,train_size=train_size/(train_size+val_size), random_state=rando)

    # train
    arr=np.full(df.Theme.shape,False)
    for t in train_themes:
        arr=np.logical_or(df.Theme==t, arr)
    train_df=df.iloc[list(arr)]

    # val
    arr=np.full(df.Theme.shape,False)
    for t in val_themes:
        arr=np.logical_or(df.Theme==t, arr)
    val_df=df.iloc[list(arr)]

    # test
    arr=np.full(df.Theme.shape,False)
    for t in test_themes:
        arr=np.logical_or(df.Theme==t, arr)
    test_df=df.iloc[list(arr)]

    return train_df, val_df, test_df
    
class SquadDataset(Dataset):
    """Custom Dataset for SQuAD data compatible with torch.utils.data.DataLoader."""

    def __init__(self, Theme, Paragraph, Question, Answer_possible, Answer_text, Answer_start):
        """Set the path for context, question and labels."""
        self.Theme=Theme
        self.Paragraph=Paragraph
        self.Question=Question
        self.Answer_possible=Answer_possible
        self.Answer_text=Answer_text
        self.Answer_start=Answer_start

    def __getitem__(self, index):
        """Returns one data tuple of the form ( word context, character context, word question,
         character question, answer)."""
        return self.Theme[index], self.Paragraph[index], self.Question[index], self.Answer_possible[index], self.Answer_text[index], self.Answer_start[index]

    def __len__(self):
        return len(self.Theme)

def create_dataset(train_df, val_df, test_df):
    train_dataset=SquadDataset(train_df.Theme, train_df.Paragraph, train_df.Question, train_df.Answer_possible, train_df.Answer_text, train_df.Answer_start)
    val_dataset=SquadDataset(val_df.Theme, val_df.Paragraph, val_df.Question, val_df.Answer_possible, val_df.Answer_text, val_df.Answer_start)
    test_dataset=SquadDataset(test_df.Theme, test_df.Paragraph, test_df.Question, test_df.Answer_possible, test_df.Answer_text, test_df.Answer_start)
    return train_dataset, val_dataset, test_dataset

def lowercase(input_text):
    # Convert to lowercase if specified
    input_text = input_text.lower()
    return input_text

def remove_urls(input_text):
    # Remove URLs if specified
    input_text = re.sub(r"http\S+", "", input_text)
    return input_text
    
def remove_punc(input_text):
    # Remove punctuation if specified
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))
    return input_text
    
def remove_html(input_text):
    # Remove HTML tags if specified
    soup = BeautifulSoup(input_text, "html.parser")
    input_text = soup.get_text()
    return input_text
    
def remove_white(input_text):
    # Remove white space if specified
    input_text = input_text.strip()
    return input_text
    
def tokenize(input_text):
    # Tokenize the input text
    tokens = nltk.word_tokenize(input_text)
    return tokens
    
def rm_stopwords(input_text, stop_words):
    # Remove stop words if specified
    tokens = nltk.word_tokenize(input_text)
    # stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens
    
def stemming(input_text, stemmer):
    # Perform stemming if specified
    tokens = nltk.word_tokenize(input_text)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens
    
def lemmatize(input_text, lemma):
    # Perform lemmatization if specified
    tokens = nltk.word_tokenize(input_text)
    tokens = [lemma.lemmatize(token) for token in tokens]
    return tokens

def preprocess(df, args_low, args_urls, args_punct, args_html, args_white, args_stop, args_stem, args_lemm):
    # performs preprocessing on dataset
    if args_low:
        df["Paragraph"]=df["Paragraph"].apply(lowercase)
        df["Question"]=df["Question"].apply(lowercase)
        
    if args_urls:
        df["Paragraph"]=df["Paragraph"].apply(remove_urls)
        df["Question"]=df["Question"].apply(remove_urls)

    if args_punct:
        df["Paragraph"]=df["Paragraph"].apply(remove_punc)
        df["Question"]=df["Question"].apply(remove_punc)

    if args_html:
        df["Paragraph"]=df["Paragraph"].apply(remove_html)
        df["Question"]=df["Question"].apply(remove_html)

    if args_white:
        df["Paragraph"]=df["Paragraph"].apply(remove_white)
        df["Question"]=df["Question"].apply(remove_white)

    if not(args_stop or args_stem or args_lemm):
        df["Paragraph"]=df["Paragraph"].apply(tokenize)
        df["Question"]=df["Question"].apply(tokenize)

    if args_stop:
        stop_words = set(stopwords.words("english"))
        df["Paragraph"]=df["Paragraph"].apply(lambda x: rm_stopwords(x,stop_words))
        df["Question"]=df["Question"].apply(lambda x: rm_stopwords(x,stop_words))

    if args_stem:
        stemmer = PorterStemmer()
        df["Paragraph"]=df["Paragraph"].apply(lambda x: stemming(x,stemmer))
        df["Question"]=df["Question"].apply(lambda x: stemming(x,stemmer))

    if args_lemm:
        lemma = WordNetLemmatizer()
        df["Paragraph"]=df["Paragraph"].apply(lambda x: lemmatize(x, lemma))
        df["Question"]=df["Question"].apply(lambda x: lemmatize(x, lemma))

    return df
    

if __name__ == "__main__":
    # Define command-line arguments using argparse
    import torch
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import numpy as np
    import random
    import os
    from efficiency.log import show_time
    import argparse
    import re
    import string
    import nltk
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    set_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=0, help="sets random state for dataset split")
    parser.add_argument("--low", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--punct", action="store_true", help="Remove punctuation from text")
    parser.add_argument("--html", action="store_true", help="Remove HTML tags from text")
    parser.add_argument("--white", action="store_true", help="Remove leading and trailing white space from text")
    parser.add_argument("--stop", action="store_true", help="Remove stop words from text")
    parser.add_argument("--stem", action="store_true", help="Perform stemming on text")
    parser.add_argument("--lemm", action="store_true", help="Perform lemmatization on text")
    parser.add_argument("--train_size", type=int, default=70, help="training set size in percent")
    parser.add_argument("--val_size", type=int, default=15, help="val set size in percent")
    parser.add_argument("--test_size", type=int, default=15, help="test set size in percent")
    args = parser.parse_args()
    
    df=read_data()
    df = preprocess(df, args.low, args.urls, args.punct, args.html, args.white, args.stop, args.stem, args.lemm)
    train_df, val_df, test_df=split(df, args.train_size, args.val_size, args.test_size, args.random)
    train_dataset, val_dataset, test_dataset=create_dataset(train_df, val_df, test_df)
    print(train_df.head())
    print(train_df.shape)
