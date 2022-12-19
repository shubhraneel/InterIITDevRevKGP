import pandas as pd 
from sklearn.model_selection import train_test_split

from utils import set_seed

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default="config.yaml", desc="Config File")

	args = parser.parse_args()
	with open(args.config) as f:
		config = yaml.safe_load(f)
		config = Config(**config)

	set_seed(config.seed)

	df = pd.read_csv(config.data.data_path)
	df_train, df_test = train_test_split(df, test_size=config.data.test_size, random_state=config.seed)
	df_train, df_val = train_test_split(df_train, test_size=config.data.test_size, random_state=config.seed)
	
	tokenizer = AutoTokenizer.from_pretrained(cofig.model.model_path, TOKENIZERS_PARALLELISM=True) # add local_files_only=local_files_only if using server

	train_ds = SQuAD_Dataset(config, df_train, tokenizer)
	val_ds = SQuAD_Dataset(config, df_val, tokenizer)
	test_ds = SQuAD_Dataset(config, df_test, tokenizer)

