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

from onnxruntime.transformers import optimizer as onnx_optimizer
import onnxruntime
import torch.onnx

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


def reformat_data_for_sqlite(df, split):

	context_doc_id = df[["context", "context_id"]].rename(
		columns={"context": "text", "context_id": "id"})
	context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])
	context_doc_id["id"] = context_doc_id["id"].astype(str)
	context_doc_id.to_json(
		"data-dir/{}/contexts.json".format(split), orient="records", lines=True)


def prepare_retriever(df, db_path, split):
	reformat_data_for_sqlite(df, f"{split}")
	if (os.path.exists(f"data-dir/{split}/{db_path}")):
		os.remove(f"data-dir/{split}/{db_path}")

	store_contents(data_path=f"data-dir/{split}/contexts.json",
				   save_path=f"data-dir/{split}/{db_path}", preprocess=None, num_workers=2)
	build_tf_idf_wrapper(db_path=f"data-dir/{split}/{db_path}",
						 out_dir=f"data-dir/{split}", ngram=3, hash_size=(2**25), num_workers=2)


if __name__ == "__main__":

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

		prepare_retriever(df_train, "sqlite_con.db", "train")
		prepare_retriever(df_val, "sqlite_con.db", "val")
		prepare_retriever(df_test, "sqlite_con.db", "test")

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
			checkpoint = torch.load("checkpoints/{}/model_optimizer.pt".format(config.load_path))
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		retriever = None
		if (config.use_drqa):
			tfidf_path = "data-dir/test/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
			questions_df = df_test[["question", "title_id"]]
			db_path = "data-dir/test/sqlite_con.db"
			retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path)

		trainer = Trainer(config=config, model=model,
						  optimizer=optimizer, device=device, tokenizer=tokenizer, ques2idx=ques2idx, retriever=retriever)

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
			test_metrics = trainer.calculate_metrics(df_test)
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
