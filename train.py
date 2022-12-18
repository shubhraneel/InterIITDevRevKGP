import pytorch_lightning as pl
from bert_retriever import BertRetriever
from bert_qa import BertQA
import argparse
from torch.optim import Adam, SGD, lr_scheduler
import yaml
from config import Config

import pickle

class RetrieverModule(pl.LightningModule):
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


class ReaderModule(pl.LightningModule):
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


def train_retriever(
    model_name="bert", save_path="models/", 
    dataloader_path="data.pkl", epochs=10, 
    context_path=None, optim_function=None, optim_params=None,
    sched_function=None, sched_params=None, model_params=None
):

    if model_name == "bert":
        model = BertRetriever(**model_params)

    with open(dataloader_path) as f:
        train_dataloader = pickle.load(f)

    with open(context_path) as f:
        context_dict = pickle.load(f)
    
    module = RetrieverModule(model=model)

    # add the actual optimizer functions based on the string input
    if optim_function=="adam":
        optim_func = Adam
    if sched_function=="reduce_lr_on_plateau":
        sched_func = lr_scheduler.ReduceLROnPlateau

    module.configure_optimizers(
        optim_function=optim_func, sched_function=sched_func, 
        optim_params=optim_params, sched_func_params=sched_func_params,
        sched_params=sched_params
    )

    trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path)
    trainer.fit(model=module, train_dataloaders=train_dataloader)


def train_reader(
    model_name="bert", save_path="models/", 
    dataloader_path="data.pkl", epochs=10,
    optim_function=None, optim_params=None,
    sched_function=None, sched_params=None,
    model_params=None
):

    if model_name == "bert":
        model = BertQA(**model_params)

    with open(dataloader_path) as f:
        train_dataloader = pickle.load(f)

    with open(context_path) as f:
        context_dict = pickle.load(f)
    
    module = ReaderModule(model=model)

    # add the actual optimizer functions based on the string input
    if optim_function=="adam":
        optim_func = Adam
    if sched_function=="reduce_lr_on_plateau":
        sched_func = lr_scheduler.ReduceLROnPlateau

    module.configure_optimizers(
        optim_function=optim_func, sched_function=sched_func, 
        optim_params=optim_params, sched_func_params=sched_func_params,
        sched_params=sched_params
    )

    trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path)
    trainer.fit(model=module, train_dataloaders=train_dataloader)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", desc="Config File")

    # parser.add_argument('--qa', action="store_true")
    # parser.add_argument('--retrieve', action="store_true")
    # parser.add_argument('--save_path')
    # parser.add_argument('--model_name')
    # parser.add_argument('--dataloader_path')
    # parser.add_argument('--epochs', type=int)
    # parser.add_argument('--optim')
    # parser.add_argument('--sched')
    # parser.add_argument('--lr', type=float)
    # parser.add_argument(
    #     '--optim_params', default="", 
    #     desc="Provide other optional optimizer params as param1=value1,param2=value2,..."
    # )
    # parser.add_argument(
    #     '--sched_params', default="", 
    #     desc="Provide other optional scheduler params as param1=value1,param2=value2,..."
    # )

    # # model specific arguments, we can do this in a different way with YAML config
    # parser.add_argument('--context_path')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
        config = Config(**config)

    if config.task == "qa":
        train_reader(
            model_name=config.model.model_name, 
            save_path=config.model.save_path, 
            dataloader_path=config.data.train_dataloader_path,
            epochs=config.training.epochs,
            model_params=config.model.params.__dict__,
            optim_function=config.training.optim_function,
            optim_params=config.training.optim_params.__dict__,
            sched_function=config.training.sched_function,
            sched_func_params=config.training.sched_func_params.__dict__,
            sched_params=config.training.sched_params.__dict__
        )

    elif config.task == "retrieve":
        train_retriever(
            model_name=config.model.model_name, 
            save_path=config.model.save_path, 
            dataloader_path=config.data.train_dataloader_path,
            epochs=config.training.epochs,
            model_params=config.model.params.__dict__,
            optim_function=config.training.optim_function,
            optim_params=config.training.optim_params.__dict__,
            sched_function=config.training.sched_function,
            sched_func_params=config.training.sched_func_params.__dict__,
            sched_params=config.training.sched_params.__dict__
        )
