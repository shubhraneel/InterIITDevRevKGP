import os
import yaml
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
# from pytorch_lightning.loggers import WandbLogger

from config import Config
from utils import set_seed
from data import SQuAD_Dataset
from src import BaselineQA
from utils import Trainer
import torch

# from pytorch_lightning.loggers import WandbLogger
import wandb
from data import SQuAD_Dataset_fewshot
from src import AutoModel_Classifier_QA, FewShotQA_Model

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config.yaml", help="Config File")

	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
		wandb.init(name=config["wandb_path"], entity="agv", project='interiit-devrev', config=config)
		config = Config(**config)

	set_seed(config.seed)

	print("Reading data csv")
	df = pd.read_csv(config.data.data_path)
	df = df.drop_duplicates(subset=["Unnamed: 0"]).drop_duplicates(subset=["Question"]).reset_index(drop=True)

	# df = pd.read_excel(config.data.data_path)
	# TODO: Split the dataset in a way where training theme question-context pair should not be
	# split into train/test/val. Keep it only in the train.
	# Mixup allowed between val and test.
	splitter = GroupShuffleSplit(train_size=config.data.train_frac, n_splits=1, random_state=config.seed)
	split = splitter.split(df, groups=df['Theme'])
	train_inds, val_inds = next(split)
	df_train = df.iloc[train_inds]
	df_val = df.iloc[val_inds]

	df_val, df_test = train_test_split(df_val, test_size=(config.data.test_frac/(config.data.test_frac + config.data.val_frac)), random_state=config.seed)
	#df_train, df_test = train_test_split(df, test_size=config.data.test_size, random_state=config.seed)
	#df_train, df_val = train_test_split(df_train, test_size=config.data.test_size, random_state=config.seed)
	tokenizer 			= AutoTokenizer.from_pretrained(config.model.model_path, TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length") # add local_files_only=local_files_only if using server

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if (config.data.apply_aliasing):
		print("Applying aliasing on train data")
		from utils.aliasing import create_alias
		df_train = create_alias(df_train)

	if config.fewshot_qa:
		train_ds = SQuAD_Dataset_fewshot(config, df_train, tokenizer, mask_token)
		val_ds = SQuAD_Dataset_fewshot(config, df_val, tokenizer, mask_token)
		test_ds = SQuAD_Dataset_fewshot(config, df_test, tokenizer, mask_token)

		train_dataloader = DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
		val_dataloader = DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
		test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

		model = FewShotQA_Model(config, tokenizer=tokenizer)

		if (config.train):
			model.__train__(train_dataloader)
		
		if (config.inference):
			model.__inference__(test_dataloader)

		qa_f1, ttime_per_example = model.few_shot_calculate_metrics(test_dataloader)
		print(f"QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
	
	else:
		model 				= BaselineQA(config, device).to(device)
		optimizer	 		= torch.optim.Adam(model.parameters(), lr=config.training.lr)

		if (config.load_model_optimizer):
			print("loading model and optimizer from checkpoints/{}/model_optimizer.pt".format(config.wandb_path))
			checkpoint = torch.load("checkpoints/{}/model_optimizer.pt".format(config.wandb_path))
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		trainer 			= Trainer(config=config, model=model, optimizer=optimizer, device=device, tokenizer=tokenizer)

		if (config.train):
			print("Creating train dataset")
			train_ds 			= SQuAD_Dataset(config, df_train, tokenizer)
			train_dataloader 	= DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
			print("length of train dataset: {}".format(train_ds.__len__()))

			print("Creating val dataset")
			val_ds 				= SQuAD_Dataset(config, df_val, tokenizer)
			val_dataloader 		= DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
			print("length of val dataset: {}".format(val_ds.__len__()))

			trainer.train(train_dataloader, val_dataloader)

		if (config.inference):
			print("Creating test dataset")
			test_ds 			= SQuAD_Dataset(config, df_test, tokenizer)
			test_dataloader 	= DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)
			print("length of test dataset: {}".format(test_ds.__len__()))
		
			# calculate_metrics(test_ds, test_dataloader, wandb_logger)
			test_metrics = trainer.calculate_metrics(test_ds, test_dataloader)
			print(test_metrics)
		
		if (config.save_model_optimizer):
			print("saving model and optimizer at checkpoints/{}/model_optimizer.pt".format(config.wandb_path))
			os.makedirs("checkpoints/{}/".format(config.wandb_path), exist_ok=True)
			torch.save({
            	'model_state_dict': model.state_dict(),
            	'optimizer_state_dict': optimizer.state_dict(),
            }, "checkpoints/{}/model_optimizer.pt".format(config.wandb_path))

		# model = AutoModel_Classifier_QA(config, tokenizer=tokenizer, logger=wandb_logger)
		# model.__train__(train_dataloader)	
		# model.__inference__(test_dataloader)
	
		# classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_dataloader)
		# print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
	

		
