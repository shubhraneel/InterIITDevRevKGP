import numpy as np
from random import choices
from data import SQuAD_Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

def boosting(trainer,config,df,val_dataloader):

    # Initial Probability distribution over the training set for sampling the data points
    D = np.ones(len(df)) / (len(df))

    # Data indices list used to sample indices of data points
    data_index = [i+1 for i in range(len(df['arg']))]

    # Store weights associated with each model
    alpha = np.zeros(5)

    # Iterate for 5 base learners
    for i in range(config.num_learners):
        sampled_indices = choices(data_index, D, k = config.data.sample_percentage*len(df))
        
        df_temp=df.iloc[sampled_indices]
        train_ds = SQuAD_Dataset(config, df_temp, trainer.tokenizer)
        train_dataloader = DataLoader(
			train_ds, batch_size=config.data.train_batch_size, collate_fn=train_ds.collate_fn)

        metrics=trainer.boosted_train(i,train_dataloader,val_dataloader)
        mean_squad_f1=np.mean(metrics)

        # Update the weight for model
        alpha[i] = 0.5 * np.log((mean_squad_f1 / (1-mean_squad_f1)))

        norm_metrics=(metrics-np.min(metrics))/(np.max(metrics)-np.min(metrics))-0.5
        D[sampled_indices]=D[sampled_indices]*norm_metrics
        D = D / np.sum(D)

    return alpha
