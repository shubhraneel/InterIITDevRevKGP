import os
import yaml
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from pytorch_lightning.loggers import WandbLogger

from config import Config
from utils import set_seed
from data import SQuAD_Dataset
from src import BaselineQA
from utils import Trainer

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config.yaml", help="Config File")

	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
		# wandb_logger = WandbLogger(name=config['wandb'], project='interiit-devrev')
		# wandb_logger.experiment.config.update(config)
		config = Config(**config)

	set_seed(config.seed)

	df = pd.read_csv(config.data.data_path)
	df = df.drop_duplicates(subset=["Unnamed: 0"])
	# df = pd.read_excel(config.data.data_path)
	# TODO: Split the dataset in a way where training theme question-context pair should not be
	# split into train/test/val. Keep it only in the train.
	# Mixup allowed between val and test.

	df_train, df_test 	= train_test_split(df, test_size=config.data.test_size, random_state=config.seed)
	df_train, df_val 	= train_test_split(df_train, test_size=config.data.test_size, random_state=config.seed)
	
	tokenizer 			= AutoTokenizer.from_pretrained(config.model.model_path, TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length") # add local_files_only=local_files_only if using server

	train_ds 			= SQuAD_Dataset(config, df_train, tokenizer)
	print("length of train dataset: {}".format(train_ds.__len__()))

	val_ds 				= SQuAD_Dataset(config, df_val, tokenizer)
	print("length of val dataset: {}".format(val_ds.__len__()))

	test_ds 			= SQuAD_Dataset(config, df_test, tokenizer)
	print("length of test dataset: {}".format(test_ds.__len__()))

	# check if datasets have been encoded properly
	# train_ds.print_row(15)
	# val_ds.print_row(3)
	# test_ds.print_row(69)
	# example = train_ds.__getitem__(12)
	# print(example["question_context_offset_mapping"])

	train_dataloader 	= DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
	val_dataloader 		= DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
	test_dataloader 	= DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = BaselineQA(config, device).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = config.training.lr)
	trainer = Trainer(config, model, optimizer, device)

	trainer.train(train_dataloader, val_dataloader)
	# calculate_metrics(test_ds, test_dataloader, wandb_logger)
	test_metrics = trainer.calculate_metrics(test_ds, test_dataloader)
	print(test_metrics)

	# model.__train__(train_dataloader, logger = wandb_logger)
	# model.__inference__(test_ds, test_dataloader, logger = wandb_logger)

	# classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_ds, test_dataloader, logger = wandb_logger)

	# print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")