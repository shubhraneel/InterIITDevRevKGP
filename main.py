import os
import yaml
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

from config import Config
from utils import set_seed
from utils import create_alias
from data import SQuAD_Dataset, SQuAD_Dataset_fewshot, SQuAD_Dataset_lstm
from pytorch_lightning.loggers import WandbLogger
from src import AutoModel_Classifier_QA, FewShotQA_Model, LSTMModel, GRUModel, StackedLSTMModel
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config.yaml", help="Config File")

	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
		wandb_logger = WandbLogger(name=config['wandb'], project='interiit-devrev')
		wandb_logger.experiment.config.update(config)
		config = Config(**config)

	set_seed(config.seed)

	# df = pd.read_excel(config.data.data_path)
	df = pd.read_csv(config.data.data_path)
	# TODO: Split the dataset in a way where training theme question-context pair should not be
	# split into train/test/val. Keep it only in the train.
	# Mixup allowed between val and test.
	splitter = GroupShuffleSplit(train_size=(1-config.data.test_size)**2,n_splits=1, random_state=config.seed)
	split = splitter.split(df,groups=df['Theme'])
	train_inds , val_inds = next(split)
	df_train = df.iloc[train_inds]
	df_val = df.iloc[val_inds]

	df_val, df_test = train_test_split(df_val, test_size = (config.data.test_size)/(config.data.test_size*(1-config.data.test_size)+config.data.test_size),
 										random_state=config.seed)
	#df_train, df_test = train_test_split(df, test_size=config.data.test_size, random_state=config.seed)
	#df_train, df_val = train_test_split(df_train, test_size=config.data.test_size, random_state=config.seed)
	
	tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, TOKENIZERS_PARALLELISM=True, model_max_length=512, padding="max_length") # add local_files_only=local_files_only if using server

	df_train_alias = create_alias(df_train,config.data.alias_flag)

	mask_token = tokenizer.mask_token

	if config.fewshot_qa:
		train_ds = SQuAD_Dataset_fewshot(config, df_train_alias, tokenizer, mask_token)
		val_ds = SQuAD_Dataset_fewshot(config, df_val, tokenizer, mask_token)
		test_ds = SQuAD_Dataset_fewshot(config, df_test, tokenizer, mask_token)

		train_dataloader = DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
		val_dataloader = DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
		test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

		model = FewShotQA_Model(config, tokenizer=tokenizer)

		model.__train__(train_dataloader)
		model.__inference__(test_dataloader)

		qa_f1, ttime_per_example = model.few_shot_calculate_metrics(test_dataloader)
		print(f"QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
	
	elif config.automodel:
		train_ds = SQuAD_Dataset(config, df_train_alias, tokenizer)
		val_ds = SQuAD_Dataset(config, df_val, tokenizer)
		test_ds = SQuAD_Dataset(config, df_test, tokenizer)

		train_dataloader = DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
		val_dataloader = DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
		test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)
		model = AutoModel_Classifier_QA(config, tokenizer=tokenizer, logger=wandb_logger)

		model.__train__(train_dataloader)
		model.__inference__(test_dataloader)
	
		classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_dataloader)
		print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")
	
	else:
		train_ds = SQuAD_Dataset_lstm(config, df_train_alias, tokenizer)
		val_ds = SQuAD_Dataset_lstm(config, df_val, tokenizer)
		test_ds = SQuAD_Dataset_lstm(config, df_test, tokenizer)

		train_dataloader = DataLoader(train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)
		val_dataloader = DataLoader(val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn)
		test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)

		# Define the model
		#model = LSTMModel(config, tokenizer=tokenizer, logger=wandb_logger)
		#model = GRUModel(config, tokenizer=tokenizer, logger=wandb_logger)
		model = StackedLSTMModel(config, tokenizer=tokenizer, logger=wandb_logger)
		
		model.__train__(train_dataloader)
		model.__inference__(test_dataloader)
		
		classification_f1, qa_f1, ttime_per_example = model.calculate_metrics(test_dataloader)
		print(f"Classification F1: {classification_f1}, QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")

	

		
