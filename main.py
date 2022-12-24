import os
import yaml
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

from config import Config
from utils import set_seed
from data import SQuAD_Dataset
from src import BaselineQA
from utils import Trainer
import torch

from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader

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

	# df = pd.read_excel(config.data.data_path)
	df = pd.read_csv(config.data.data_path)
	# TODO: Split the dataset in a way where training theme question-context pair should not be
	# split into train/test/val. Keep it only in the train.
	# Mixup allowed between val and test.

	df_train, df_test = train_test_split(df, test_size=config.data.test_size, random_state=config.seed)
	df_train, df_val = train_test_split(df_train, test_size=config.data.test_size, random_state=config.seed)
	
	tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length") # add local_files_only=local_files_only if using server

	train_ds = SQuAD_Dataset(config, df_train, tokenizer, train = True)
	# val_ds = SQuAD_Dataset(config, df_val, tokenizer, train = False)
	# test_ds = SQuAD_Dataset(config, df_test, tokenizer, train = False)

	train_dataloader = DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
	# val_dataloader = DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
	# test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"

	model = BaselineQA(config, device).to(device)
	# model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = config.training.lr)
	trainer = Trainer(config, model, optimizer, device)

	trainer.train(train_dataloader)

	# model.__train__(train_dataloader, logger = wandb_logger)
	# model.__inference__(test_ds, test_dataloader, logger = wandb_logger)

	# classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_ds, test_dataloader, logger = wandb_logger)

	# print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
# 