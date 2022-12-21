# -*- coding: utf-8 -*-
"""Copy of CNLP_Task2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EZR2ExMRK6tacnar66ALtAeofqgOPGrh
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
import nltk
nltk.download('punkt')

df = pd.read_csv('train_data.csv', encoding = 'utf-8')

from nltk.tokenize import sent_tokenize

answer = df.loc[1].at["Answer_text"][2:-2]
print(answer)

def sent_index(text, para, ans_pos):
  if ans_pos == False:
    return -1
  ans = text[2:-2]
  sents = sent_tokenize(para)
  for sent in sents:
    if ans in sent:
      return (sents.index(sent))

df['Sentence Index'] = df.apply(lambda row : sent_index(row['Answer_text'], row['Paragraph'], row['Answer_possible']), axis = 1)

df.to_csv('train_data_sentidx.csv')
del df
#torch.cuda.empty_cache()
#%reset

dataframe = pd.read_csv('train_data_sentidx.csv')

def sentence_tokenize(text):
  return sent_tokenize(text)

dataframe['Paragraph']= dataframe['Paragraph'].apply(lambda x:sentence_tokenize(x))
dataframe['Question'] = dataframe['Question'].apply(lambda x:sentence_tokenize(x))

import string

def remove_punctuation(sents):
  res = []
  for text in sents:
    no_punc="".join([i for i in text if i not in string.punctuation])
    res.append(no_punc)
  return res

def remove_punctuation2(text):
    no_punc="".join([i for i in text if i not in string.punctuation])
    return no_punc

dataframe['Paragraph']= dataframe['Paragraph'].apply(lambda x:remove_punctuation(x))
dataframe['Question'] = dataframe['Question'].apply(lambda x:remove_punctuation(x))
dataframe['Answer_text'] = dataframe['Answer_text'].apply(lambda x:remove_punctuation2(x))

def lower_func(sents):
  res = []
  for text in sents:
    sent = text.lower()
    res.append(sent)
  return res


dataframe['Paragraph']= dataframe['Paragraph'].apply(lambda x: lower_func(x))
dataframe['Question']= dataframe['Question'].apply(lambda x: lower_func(x))
dataframe['Answer_text']= dataframe['Answer_text'].apply(lambda x: x.lower())

'''!mkdir GloVe
!curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
!unzip GloVe/glove.840B.300d.zip -d GloVe/
!mkdir fastText
!curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
!unzip fastText/crawl-300d-2M.vec.zip -d fastText/

!mkdir encoder
!curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
!curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl'''

# Commented out IPython magic to ensure Python compatibility.
# import stuff
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from random import randint

import numpy as np
import torch

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

use_cuda = True
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=300000)

embeddings_all = []
for para in dataframe['Paragraph']:
  embeddings = model.encode(para, bsize=128, tokenize=False, verbose=True)
  embeddings_all.append(embeddings)

embeddings_array = np.array(embeddings_all)

del embeddings_all
#torch.cuda.empty_cache()

qs_emb = []
for para in dataframe['Question']:
  embeddings = model.encode(para, bsize=128, tokenize=False, verbose=True)
  qs_emb.append(embeddings)

qs_emb_array = np.array(qs_emb)

del qs_emb
#torch.cuda.empty_cache()

del dataframe
#torch.cuda.empty_cache()

#np.savez("sent_embs", embeddings_array)

#np.savez("qs_embs", qs_emb_array)

import sklearn
from sklearn.metrics.pairwise import cosine_similarity

cosine_sims = []
for i in range(len(embeddings_array)):
  temp = []
  for sent in embeddings_array[i]:
    temp.append(cosine_similarity(sent.reshape(1, -1), qs_emb_array[i][0].reshape(1, -1)))
  cosine_sims.append(np.array(temp))

np.savez("cosine_similarities", np.array(cosine_sims))

del embeddings_array
del qs_emb_array
del cosine_sims
#torch.cuda.empty_cache()

# Commented out IPython magic to ensure Python compatibility.
# %reset