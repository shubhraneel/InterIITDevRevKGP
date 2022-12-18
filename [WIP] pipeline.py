# imports
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import argparse
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# read from csv
df=pd.read_csv("train_data.csv")
df=df.drop(labels=["Unnamed: 0"],axis="columns")

# themes
themes=df["Theme"].unique()

# train_val_test split
train_themes,test_themes=train_test_split(themes,train_size=(train_size+val_size)/100)
train_themes,val_themes=train_test_split(themes,train_size=train_size/(train_size+val_size))


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

train_dataset=SquadDataset(train_df.Theme, train_df.Paragraph, train_df.Question, train_df.Answer_possible, train_df.Answer_text, train_df.Answer_start)
val_dataset=SquadDataset(val_df.Theme, val_df.Paragraph, val_df.Question, val_df.Answer_possible, val_df.Answer_text, val_df.Answer_start)
test_dataset=SquadDataset(test_df.Theme, test_df.Paragraph, test_df.Question, test_df.Answer_possible, test_df.Answer_text, test_df.Answer_start)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def preprocess(input_text, lowercase, remove_urls, remove_punctuation, remove_html, remove_whitespace, remove_stopwords, stem, lemmatize):
    # Convert to lowercase if specified
    if lowercase:
        input_text = input_text.lower()
    
    # Remove URLs if specified
    if remove_urls:
        input_text = re.sub(r"http\S+", "", input_text)
    
    # Remove punctuation if specified
    if remove_punctuation:
        input_text = input_text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove HTML tags if specified
    if remove_html:
        soup = BeautifulSoup(input_text, "html.parser")
        input_text = soup.get_text()
    
    # Remove white space if specified
    if remove_whitespace:
        input_text = input_text.strip()
    
    # Tokenize the input text
    tokens = nltk.word_tokenize(input_text)
    
    # Remove stop words if specified
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Perform stemming if specified
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Perform lemmatization if specified
    if lemmatize:
        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(token) for token in tokens]
    
    # Perform other preprocessing steps
    # ...
    
    return tokens

if __name__ == "__main__":
    # Define command-line arguments using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowercase", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--remove_urls", action="store_true", help="Remove URLs from text")
    parser.add_argument("--remove_punctuation", action="store_true", help="Remove punctuation from text")
    parser.add_argument("--remove_html", action="store_true", help="Remove HTML tags from text")
    parser.add_argument("--remove_whitespace", action="store_true", help="Remove leading and trailing white space from text")
    parser.add_argument("--remove_stopwords", action="store_true", help="Remove stop words from text")
    parser.add_argument("--stem", action="store_true", help="Perform stemming on text")
    parser.add_argument("--lemmatize", action="store_true", help="Perform lemmatization on text")
    parser.add_argument("--train_size", action="store_true", help="training set size in percent")
    parser.add_argument("--val_size", action="store_true", help="val set size in percent")
    parser.add_argument("--test_size", action="store_true", help="test set size in percent")
    # parser.add_argument("input_text", help="Text to be preprocessed")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for preprocessing')
    args = parser.parse_args()
    
    # Preprocess the input text
#     output_text = preprocess(args.input_text, args.lowercase, args.remove_urls, args.remove_punctuation, args.remove_html, 
#                              args.remove_whitespace, args.remove_stopwords, args.stem, args.lemmatize)
    
    # Print the output text
    # print(output_text)
