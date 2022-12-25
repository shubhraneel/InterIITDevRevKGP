import spacy
from spacy import displacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

NER = spacy.load("en_core_web_sm")

# Run the following commented commands on the terminal to download InferSent packages
'''!mkdir GloVe
!curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
!unzip GloVe/glove.840B.300d.zip -d GloVe/
!mkdir fastText
!curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
!unzip fastText/crawl-300d-2M.vec.zip -d fastText/

!mkdir encoder
!curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
!curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl'''

#!pip install transformers

# import stuff
%load_ext autoreload
%autoreload 2
%matplotlib inline

from random import randint

import numpy as np
import torch

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model2 = InferSent(params_model)
model2.load_state_dict(torch.load(MODEL_PATH))

use_cuda = False
model2 = model2.cuda() if use_cuda else model2

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model2.set_w2v_path(W2V_PATH)

model2.build_vocab_k_words(K=200000)

# Load model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering
model_name = "aychang/bert-base-cased-trec-coarse"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use pipeline
from transformers import pipeline
nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer1 = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model1 = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp2 = pipeline("ner", model=model1, tokenizer=tokenizer1)

def get_ans(question, sentence):
  question_type = nlp(question)[0]['label']
  ner_results = nlp2(sentence)
  ner_results2 = NER(sentence)

  per = []
  loc = []
  misc = []
  org = []
  num = []
  per_str = ""
  loc_str =""
  misc_str=""
  org_str=""
  num_str = ""

  for word in ner_results2.ents:
      if word.label_ == 'DATE' or word.label_ == 'CARDINAL' or word.label_=='MONEY' or word.label_=='PERCENT' or word.label_=='QUANTITY' or word.label_=='TIME':
        num.append(word.text)

  for i in range(len(ner_results)):
    if ner_results[i]['entity'] == 'B-LOC' or ner_results[i]['entity'] == 'I-LOC' or ner_results[i]['entity'] == 'O-LOC':
      loc_str = loc_str + ner_results[i]['word'] + " "
      if (i == len(ner_results)-1) or ((i+1)<len(ner_results) and ((ner_results[i+1]['entity'] !='I-LOC' and ner_results[i+1]['entity'] != 'O-LOC' and ner_results[i+1]['entity'] != 'B-LOC'))):
        loc.append(loc_str[:-1])
        loc_str = ''

    if ner_results[i]['entity'] == 'B-PER' or ner_results[i]['entity'] == 'I-PER' or ner_results[i]['entity'] == 'O-PER':
      per_str = per_str + ner_results[i]['word'] + " "
      if (i == len(ner_results)-1) or ((i+1)<len(ner_results) and ((ner_results[i+1]['entity'] !='I-PER' and ner_results[i+1]['entity'] != 'O-PER' and ner_results[i+1]['entity'] != 'B-PER'))):
        per.append(per_str[:-1])
        per_str = ''
      
    if ner_results[i]['entity'] == 'B-ORG' or ner_results[i]['entity'] == 'I-ORG' or ner_results[i]['entity'] == 'O-ORG':
      org_str = org_str + ner_results[i]['word'] + " "
      if (i == len(ner_results)-1) or ((i+1)<len(ner_results) and ((ner_results[i+1]['entity'] !='I-ORG' and ner_results[i+1]['entity'] != 'O-ORG' and ner_results[i+1]['entity'] != 'B-ORG'))):
        org.append(org_str[:-1])
        org_str = ''

    if ner_results[i]['entity'] == 'B-MISC' or ner_results[i]['entity'] == 'I-MISC' or ner_results[i]['entity'] == 'O-MISC':
      misc_str = misc_str + ner_results[i]['word'] + " "
      if (i == len(ner_results)-1) or ((i+1)<len(ner_results) and ((ner_results[i+1]['entity'] !='I-MISC' and ner_results[i+1]['entity'] != 'O-MISC' and ner_results[i+1]['entity'] != 'B-MISC'))):
        misc.append(misc_str[:-1])
        misc_str = ''

  if question_type == 'LOC':
    if len(loc) == 0:
      return []
    return np.array(loc)
  if question_type == 'HUM':
    if len(per) == 0:
      return []
    return np.array(per)
  if question_type == 'ENTY':
    if len(org) == 0 and len(misc)==0 and len(per)==0 and len(loc)==0:
      return []
    return np.concatenate((org, misc, per, loc), axis=None)
  if question_type == 'ABBR':
    if len(org)==0 and len(misc)==0 and len(loc)==0:
      return []
    return np.concatenate((org, misc, loc), axis=None)
  if question_type == 'NUM':
    if len(misc)==0 and len(num)==0:
      return []
    return np.concatenate((misc, num), axis=None)
  if question_type == 'DESC':
    if len(misc)==0:
      return []
    return misc

def ranking(question, ans_list):
  if len(ans_list)==0:
    return []
  maxim = ""
  sec_max=""
  question_emb = model2.encode([question], bsize=128, tokenize=False, verbose=False)
  ans_embs = model2.encode(ans_list, bsize=128, tokenize=False, verbose=False)
  
  cos = []
  dictionary = {}
  for i in range(len(ans_embs)):
    x = cosine_similarity(ans_embs[i].reshape(1,-1), question_emb)[0][0]
    cos.append(x)
    dictionary[ans_list[i]] = x

  if len(dictionary)!=0:
    maxim = max(dictionary, key=dictionary.get)
    dictionary.pop(maxim)
  if len(dictionary)!=0:
    sec_max = max(dictionary, key=dictionary.get)
  
  return [maxim, sec_max]

# main end-to-end function to take in the question and the candidate sentence and return two best candidates for exact answer text spans.
def exact_ans(question, sentence):
  ans_list = get_ans(question, sentence)
  exact_answer = ranking(question, ans_list)
  return exact_answer
