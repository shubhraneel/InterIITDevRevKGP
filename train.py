import pytorch_lightning as pl
from bert_retriever import BertRetriever
from bert_qa import BertQA
import argparse
from torch.optim import Adam, SGD, lr_scheduler
import yaml
from config import Config
import torch
import pickle

# class basemodel():
#     def __init__(self):
#         pass
    
#     def train(self):
#         raise NotImplementedError("No training method implemented")
    
#     def evaluate(self):
#         raise NotImplementedError("No evaluation method implemented")


# class RetrieverModule(pl.LightningModule):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
        
#     def forward(self, x):
#         output, loss = self.model(x)
#         return output, loss

#     def training_step(self, batch, batch_idx):
#         _, loss = self(batch)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         _, loss = self(batch)
#         self.log("val_loss", loss)
#         return loss

#     def configure_optimizers(self, optim_function, sched_function, optim_params, sched_func_params, sched_params):
#         optimizer = optim_function(self.parameters(), **optim_params)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": sched_function(optimizer, **sched_func_params),
#                 **sched_params
#             }
#         }

#     def predict_dataset(self, dataloader):
#         all_outputs = []
#         for batch in dataloader:
#             outputs, _ = self.model(batch)
#             all_outputs.append(self.model.decode(outputs))
#         all_outputs = torch.cat(all_outputs)
#         return all_outputs







# class basic_twostep_model(basemodel):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
    
#     def train(self):
#         if self.config.training.train_retriever:
#             self.train_reader(
#             model_name=self.config.model.model_name, 
#             save_path=self.config.model.save_path, 
#             dataloader_path=self.config.data.train_dataloader_path,
#             epochs=self.config.training.epochs,
#             model_params=self.config.model.params.__dict__,
#             optim_function=self.config.training.optim_function,
#             optim_params=self.config.training.optim_params.__dict__,
#             sched_function=self.config.training.sched_function,
#             sched_func_params=self.config.training.sched_func_params.__dict__,
#             sched_params=self.config.training.sched_params.__dict__
#         )


#         if self.config.training.train_reader:
#             self.train_retriever(
#             model_name=self.config.model.model_name, 
#             save_path=self.config.model.save_path, 
#             dataloader_path=self.config.data.train_dataloader_path,
#             epochs=self.config.training.epochs,
#             model_params=self.config.model.params.__dict__,
#             optim_function=self.config.training.optim_function,
#             optim_params=self.config.training.optim_params.__dict__,
#             sched_function=self.config.training.sched_function,
#             sched_func_params=self.config.training.sched_func_params.__dict__,
#             sched_params=self.config.training.sched_params.__dict__
#         )


#     def train_retriever(
#         self, model_name="bert", save_path="models/", 
#         dataloader_path="data.pkl", epochs=10, 
#         context_path=None, optim_function=None, optim_params=None,
#         sched_function=None, sched_params=None, model_params=None
#     ):

#         if model_name == "bert":
#             model = BertRetriever(**model_params)

#         with open(dataloader_path) as f:
#             train_dataloader = pickle.load(f)

#         with open(context_path) as f:
#             context_dict = pickle.load(f)
        
#         self.retriever = RetrieverModule(model=model)

#         # add the actual optimizer functions based on the string input
#         if optim_function=="adam":
#             optim_func = Adam
#         if sched_function=="reduce_lr_on_plateau":
#             sched_func = lr_scheduler.ReduceLROnPlateau

#         self.retriever.configure_optimizers(
#             optim_function=optim_func, sched_function=sched_func, 
#             optim_params=optim_params, sched_func_params=sched_func_params,
#             sched_params=sched_params
#         )

#         trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path)
#         trainer.fit(model=self.retriever, train_dataloaders=train_dataloader)


#     def train_reader(
#         self, model_name="bert", save_path="models/", 
#         dataloader_path="data.pkl", epochs=10,
#         optim_function=None, optim_params=None,
#         sched_function=None, sched_params=None,
#         model_params=None
#     ):

#         if model_name == "bert":
#             model = BertQA(**model_params)

#         with open(dataloader_path) as f:
#             train_dataloader = pickle.load(f)

#         with open(context_path) as f:
#             context_dict = pickle.load(f)
        
#         self.reader = ReaderModule(model=model)

#         # add the actual optimizer functions based on the string input
#         if optim_function=="adam":
#             optim_func = Adam
#         if sched_function=="reduce_lr_on_plateau":
#             sched_func = lr_scheduler.ReduceLROnPlateau

#         self.reader.configure_optimizers(
#             optim_function=optim_func, sched_function=sched_func, 
#             optim_params=optim_params, sched_func_params=sched_func_params,
#             sched_params=sched_params
#         )

#         trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path)
#         trainer.fit(model=self.reader, train_dataloaders=train_dataloader)

#     def evaluate(self, model_ckpt):
#         # @shubhraneel, sing self.reader and self.retreiver, implement the logic and return the following list:
#         # [Predicted tf, gold_tf, predicted_spans, true_spans]

#         test_dataloader


class PlModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, loss = self.model(x)
        return output, loss

    def training_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self, optim_function, sched_function, optim_params, sched_func_params, sched_params):
        optimizer = optim_function(self.parameters(), **optim_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched_function(optimizer, **sched_func_params),
                **sched_params
            }
        }

    def predict_dataset(self, dataloader):
        all_outputs = []
        for batch in dataloader:
            all_outputs.append(self.model.predict_dataset(batch))
            # outputs, _ = self.model(batch)
            # all_outputs.append(self.model.decode(outputs))
        all_outputs = torch.cat(all_outputs)
        return all_outputs


# this function chooses the model to train based on the task and model_name in the config
def choose_model(config):
    model_params = config.model.params.__dict__
    model_name = config.model.model_name
    task = config.task
    if task == 'qa':
        # has models that only do QA
        if model_name == 'bert':
            model = BertQA(**model_params)
        # any other model names go in elif
    elif task == "retrieve":
        if model_name == "retrieve":
            model = BertRetriever(**model_params)
    # any other task like question generation can go here
    return model

def train_model(config):

    model = choose_model(config)

    with open(config.data.train_dataloader_path) as f:
        train_dataloader = pickle.load(f)
    
    module = PlModule(model=model)

    # add the actual optimizer functions based on the string input
    if config.training.optim_function=="adam":
        optim_func = Adam
    if config.training.sched_function=="reduce_lr_on_plateau":
        sched_func = lr_scheduler.ReduceLROnPlateau

    module.configure_optimizers(
        optim_function=optim_func, sched_function=sched_func, 
        optim_params=config.training.optim_params, sched_func_params=config.training.sched_func_params,
        sched_params=config.training.sched_params
    )

    trainer = pl.Trainer(max_epochs=config.training.epochs, default_root_dir=config.model.save_path)
    trainer.fit(model=module, train_dataloaders=config.training.train_dataloader)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", desc="Config File")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
        config = Config(**config)

    train_model(config)
