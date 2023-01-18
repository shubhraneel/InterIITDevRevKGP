import os
import sys
import json
import yaml
import wandb
import pickle
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from config import Config
from src import BaselineQA, FewShotQA_Model
from utils import Trainer, set_seed, Retriever
from data import SQuAD_Dataset, SQuAD_Dataset_fewshot
from utils import build_tf_idf_wrapper, store_contents
from nltk.tokenize import sent_tokenize
import nltk  
from tqdm import tqdm

from onnxruntime.transformers import optimizer as onnx_optimizer
import onnxruntime
import torch.onnx

def sent_index(text, para, ans_pos):
	if ans_pos == False:
		return -1
	ans = text[2:-2]
	sents = sent_tokenize(para)
	for sent in sents:
		if ans in sent:
			return (sents.index(sent))


def load_mappings():
	with open("data-dir/con_idx_2_title_idx.pkl", "rb") as f:
		con_idx_2_title_idx = pickle.load(f)

	with open("data-dir/ques2idx.pkl", "rb") as f:
		ques2idx = pickle.load(f)

	with open("data-dir/idx2ques.pkl", "rb") as f:
		idx2ques = pickle.load(f)

	with open("data-dir/con2idx.pkl", "rb") as f:
		con2idx = pickle.load(f)

	with open("data-dir/idx2con.pkl", "rb") as f:
		idx2con = pickle.load(f)

	with open("data-dir/title2idx.pkl", "rb") as f:
		title2idx = pickle.load(f)

	with open("data-dir/idx2title.pkl", "rb") as f:
		idx2title = pickle.load(f)

		return con_idx_2_title_idx, ques2idx, idx2ques, con2idx, idx2con, title2idx, idx2title


def reformat_data_for_sqlite(df, split, drqa_mode):
	if config.drqa_mode == "sentence":
		context_doc_id_original = df[["context", "context_id"]].rename(
			columns={"context": "text", "context_id": "id"})
		context_doc_id_original = context_doc_id_original.drop_duplicates(subset=["text", "id"])
		# context_doc_id=pd.DataFrame()
		sent_list_all = []
		sent_id_all = []
		for idx, row in tqdm(context_doc_id_original.iterrows()):
			context=row['text']
			context_id=row['id']
			# print(context_id,context)
			# tokenizing to sentences
			sent_list=list(sent_tokenize(context))
			# id of the form "<c_id>_<i>" for ith sentence in c_id context
			sent_id_list= [f"{context_id}_{x}" for x in range(len(sent_list))]
			sent_list_all.extend(sent_list)
			sent_id_all.extend(sent_id_list)
			# df_local=pd.DataFrame(list(zip(sent_list,sent_id_list)),columns=['text','id'])
			# df_local=df_local.dropna()
			# # dataframe with sentence and sentence id
			# context_doc_id=pd.concat([context_doc_id,df_local],axis=0)
			# TODO: do not concat df
		context_doc_id = pd.Dataframe.from_dict({
			"text": sent_list_all,
			"id": sent_id_all
		})
		print(context_doc_id.head())
		context_doc_id = context_doc_id.dropna()
		context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])	
		context_doc_id.to_json(
			"data-dir/{}/contexts.json".format(split), orient="records", lines=True)
	elif config.drqa_mode == "both":
		context_doc_id_original = df[["context", "context_id"]].rename(
			columns={"context": "text", "context_id": "id"})
		context_doc_id_original = context_doc_id_original.drop_duplicates(subset=["text", "id"])
		# context_doc_id=pd.DataFrame()
		sent_list_all = []
		sent_id_all = []
		con_id_all = []
		for idx, row in tqdm(context_doc_id_original.iterrows()):
			context=row['text']
			context_id=row['id']
			# print(context_id,context)
			# tokenizing to sentences
			sent_list=list(sent_tokenize(context))
			# id of the form "<c_id>_<i>" for ith sentence in c_id context
			sent_id_list= [f"{context_id}_{x}" for x in range(len(sent_list))]
			con_id_list = [context_id for _ in range(len(sent_list))]
			sent_list_all.extend(sent_list)
			sent_id_all.extend(sent_id_list)
			con_id_all.extend(con_id_list)
			# df_local=pd.DataFrame(list(zip(sent_list,sent_id_list)),columns=['text','id'])
			# df_local=df_local.dropna()
			# # dataframe with sentence and sentence id
			# context_doc_id=pd.concat([context_doc_id,df_local],axis=0)
			# TODO: do not concat df
		context_doc_id = pd.Dataframe.from_dict({
			"text": sent_list_all,
			"id": sent_id_all,
			"con_id": con_id_all
		})
		print(context_doc_id.head())
		context_doc_id = context_doc_id.dropna()
		context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])	
		context_doc_id['sent_id'] = context_doc_id.index
		con_id_2_sent_ids = {}
		for i in set(context_doc_id['con_id']):
			con_id_2_sent_ids[i] = list(context_doc_id[context_doc_id['con_id'] == i]['sent_id'])
		with open("data-dir/{}/con_id_2_sent_ids.json".format(split), "w+") as f:
			json.dump(f, con_id_2_sent_ids)
		context_doc_id_original["id"] = context_doc_id_original["id"].astype(str)
		context_doc_id_original.to_json(
			"data-dir/{}/contexts.json".format(split), orient="records", lines=True)
		context_doc_id[["text", "id"]].to_json(
			"data-dir/{}/sentences.json".format(split), orient="records", lines=True)
	else:
		context_doc_id = df[["context", "context_id"]].rename(
			columns={"context": "text", "context_id": "id"})
		context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])
		context_doc_id["id"] = context_doc_id["id"].astype(str)
	# saving the data in json
		context_doc_id.to_json(
			"data-dir/{}/contexts.json".format(split), orient="records", lines=True)


def prepare_retriever(df, db_path, split, drqa_mode):
	# reformats and saves the df
	reformat_data_for_sqlite(df, f"{split}", drqa_mode)
	if (os.path.exists(f"data-dir/{split}/{db_path}")):
		os.remove(f"data-dir/{split}/{db_path}")
	# creates database from stored json
	store_contents(data_path=f"data-dir/{split}/contexts.json",
				   save_path=f"data-dir/{split}/{db_path}", preprocess=None, num_workers=2)
	build_tf_idf_wrapper(db_path=f"data-dir/{split}/{db_path}",
						 out_dir=f"data-dir/{split}", ngram=3, hash_size=(2**25), num_workers=2)
	if drqa_mode == "both":
		store_contents(data_path=f"data-dir/{split}/sentences.json",
				   save_path=f"data-dir/{split}/sen_{db_path}", preprocess=None, num_workers=2)
		build_tf_idf_wrapper(db_path=f"data-dir/{split}/sen_{db_path}",
							out_dir=f"data-dir/{split}", ngram=3, hash_size=(2**25), num_workers=2)

	# TODO: build tf-idf for both context and sentence
	# store context id to sentence id mapping in metadata


if __name__ == "__main__":
	nltk.download('punkt')
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config.yaml", help="Config File")

	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
		wandb.init(name=config["wandb_path"], entity="agv",
				   project='interiit-devrev', config=config)
		config = Config(**config)

	set_seed(config.seed)

	# TODO Explore pytorch.quantization also
	assert not config.quantize or config.ONNX, "Quantizing without ONNX Runtime is not supported"

	print("Reading data csv")
	df_train = pd.read_pickle(config.data.train_data_path)
	df_val = pd.read_pickle(config.data.val_data_path)
	df_test = pd.read_pickle(config.data.test_data_path)


	con_idx_2_title_idx, ques2idx, idx2ques, con2idx, idx2con, title2idx, idx2title = load_mappings()

	if (config.use_drqa and config.create_drqa_tfidf):
		print("using drqa")
		prepare_retriever(df_val, "sqlite_con.db", "val", config.drqa_mode)
		# prepare_retriever(df_train, "sqlite_con.db", "train",False)
		prepare_retriever(df_test, "sqlite_con.db", "test", config.drqa_mode)

	# add local_files_only=local_files_only if using server
	tokenizer = AutoTokenizer.from_pretrained(
		config.model.model_path, TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length")

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if (config.data.apply_aliasing):
		print("Applying aliasing on train data")
		from utils.aliasing import create_alias
		df_train = create_alias(df_train)

	if config.fewshot_qa:
		train_ds = SQuAD_Dataset_fewshot(
			config, df_train, tokenizer, mask_token)
		val_ds = SQuAD_Dataset_fewshot(config, df_val, tokenizer, mask_token)
		test_ds = SQuAD_Dataset_fewshot(config, df_test, tokenizer, mask_token)

		train_dataloader = DataLoader(
			train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
		val_dataloader = DataLoader(
			val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
		test_dataloader = DataLoader(
			test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

		model = FewShotQA_Model(config, tokenizer=tokenizer)

		if (config.train):
			model.__train__(train_dataloader)

		if (config.inference):
			model.__inference__(test_dataloader)

		qa_f1, ttime_per_example = model.few_shot_calculate_metrics(
			test_dataloader)
		print(
			f"QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")

	else:
		model = BaselineQA(config, device).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

		if (config.load_model_optimizer):
			print("loading model and optimizer from checkpoints/{}/model_optimizer.pt".format(config.load_path))
			checkpoint = torch.load("checkpoints/{}/model_optimizer.pt".format(config.load_path),
						map_location=torch.device(device))
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		retriever = None
		if (config.use_drqa):
			df_test['sentence_index'] = df_test.apply(lambda row : sent_index(row['answer_text'], row['context'], row['answerable']), axis = 1).fillna(0)
			df_train['sentence_index'] = df_train.apply(lambda row : sent_index(row['answer_text'], row['context'], row['answerable']), axis = 1).fillna(0)
			df_val['sentence_index'] = df_val.apply(lambda row : sent_index(row['answer_text'], row['context'], row['answerable']), axis = 1).fillna(0)
			
			if config.drqa_mode == 'sentence' or config.drqa_mode == 'paragraph':
				tfidf_path = "data-dir/test/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_test[["question", "title_id","sentence_index"]]
				db_path = "data-dir/test/sqlite_con.db"
				test_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=config.sentence_level)
		
				tfidf_path = "data-dir/val/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_val[["question", "title_id","sentence_index"]]
				db_path = "data-dir/val/sqlite_con.db"
				val_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=config.sentence_level)
			elif config.drqa_mode == "both":
				tfidf_path = "data-dir/test/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_test[["question", "title_id","sentence_index"]]
				db_path = "data-dir/test/sqlite_con.db"
				test_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=False)
		
				tfidf_path = "data-dir/val/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_val[["question", "title_id","sentence_index"]]
				db_path = "data-dir/val/sqlite_con.db"
				val_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=False)

				tfidf_path = "data-dir/test/sen_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_test[["question", "title_id","sentence_index"]]
				db_path = "data-dir/test/sen_sqlite_con.db"
				test_retriever_sent = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=True)
		
				tfidf_path = "data-dir/val/sen_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
				questions_df = df_val[["question", "title_id","sentence_index"]]
				db_path = "data-dir/val/sen_sqlite_con.db"
				val_retriever_sent = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=True)

		if (config.save_model_optimizer):
			print("saving model and optimizer at checkpoints/{}/model_optimizer.pt".format(config.load_path))
			os.makedirs("checkpoints/{}/".format(config.load_path), exist_ok=True)
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
			}, "checkpoints/{}/model_optimizer.pt".format(config.load_path))

		trainer = Trainer(config=config, model=model,
						  optimizer=optimizer, device=device, tokenizer=tokenizer, ques2idx=ques2idx, 
			  val_retriever=val_retriever,df_val=df_val)

		if (config.train):
			print("Creating train dataset")
			train_ds = SQuAD_Dataset(config, df_train, tokenizer)
			train_dataloader = DataLoader(
				train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
			print("length of train dataset: {}".format(train_ds.__len__()))

			print("Creating val dataset")
			val_ds = SQuAD_Dataset(config, df_val, tokenizer)
			val_dataloader = DataLoader(
				val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
			print("length of val dataset: {}".format(val_ds.__len__()))

			trainer.train(train_dataloader, val_dataloader)

	if (config.save_model_optimizer):
		print("saving model and optimizer at checkpoints/{}/model_optimizer.pt".format(config.load_path))
		os.makedirs("checkpoints/{}/".format(config.load_path), exist_ok=True)
		torch.save({
		  'model_state_dict': model.state_dict(),
		  'optimizer_state_dict': optimizer.state_dict(),
		}, "checkpoints/{}/model_optimizer.pt".format(config.load_path))


		if (config.inference):
			# print("Creating test dataset")
			# test_ds = SQuAD_Dataset(config, df_test, tokenizer)
			# test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)
			# print("length of test dataset: {}".format(test_ds.__len__()))

			# calculate_metrics(test_ds, test_dataloader, wandb_logger)
			# test_metrics = trainer.calculate_metrics(test_ds, test_dataloader)
			model.to(config.inference_device)
			if config.drqa_mode == "both":
				test_metrics = trainer.calculate_metrics(df_test,[test_retriever, test_retriever_sent],'test',config.inference_device,do_prepare=True)
			else:
				test_metrics = trainer.calculate_metrics(df_test,test_retriever,'test',config.inference_device,do_prepare=True)
			print(test_metrics)


		# model = AutoModel_Classifier_QA(config, tokenizer=tokenizer, logger=wandb_logger)
		# model.__train__(train_dataloader)
		# model.__inference__(test_dataloader)

		# classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_dataloader)
		# print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
		# tfidf_path = "data-dir/sqlite_para-tfidf-ngram=2-hash=16777216-tokenizer=corenlp.npz"
		# questions_path = "data-dir/questions_only.csv"
		# para2id_path = 'data-dir/para_title.json'
		# db_path = "data-dir/sqlite_para.db"
