import pytorch_lightning as pl
from bert_retriever import BertRetriever
from bert_qa import BertQA
import argparse

import pickle

class RetrieverModule(pl.LightningModule):
    def __init__(self, model, context_path):
        super().__init__()
        self.model = model
        with open(context_path) as f:
            self.context_dict = pickle.load(f)
        
    def forward(self, input_ids, context_id, context_name):
        output, loss = self.model(input_ids, context_id, self.context_dict[context_name])
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, context_id = batch
        loss = self(input_ids, context_id)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, context_id, context_name = batch
        loss = self(input_ids, context_id)
        self.log("val_loss", loss)
        return loss


class ReaderModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output, loss = self.model(input)
        return loss

    def training_step(self, batch, batch_idx):
        output, loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, loss = self(batch)
        self.log("val_loss", loss)
        return loss


def train_retriever(
    model_name="bert", 
    save_path="model.pt", 
    dataloader_path="data.pkl", 
    epochs=10, 
    context_path=None
):

    if model_name == "bert":
        model = BertRetriever()

    with open(dataloader_path) as f:
        train_dataloader = pickle.load(f)

    with open(context_path) as f:
        context_dict = pickle.load(f)
    
    module = RetrieverModule(model=model, context_path=context_path)

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model=module, train_dataloaders=train_dataloader)


def train_reader(model_name="bert", save_path="model.pt", dataloader_path="data.pkl", epochs=10):

    if model_name == "bert":
        model = BertQA()

    with open(dataloader_path) as f:
        train_dataloader = pickle.load(f)

    with open(context_path) as f:
        context_dict = pickle.load(f)
    
    module = ReaderModule(model=model)

    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model=module, train_dataloaders=train_dataloader)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--qa', action="store_true")
    parser.add_argument('--retrieve', action="store_true")
    parser.add_argument('--save_path')
    parser.add_argument('--model_name')
    parser.add_argument('--dataloader_path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--context_path')

    args = parser.parse_args()

    if args.qa:
        train_reader(
            model_name=args.model_name, 
            save_path=args.save_path, 
            dataloader_path=args.dataloader_path,
            epochs=args.epochs
        )

    elif args.retrieve:
        train_retriever(
            model_name=args.model_name, 
            save_path=args.save_path, 
            dataloader_path=args.dataloader_path,
            epochs=args.epochs,
            context_path=args.context_path
        )
