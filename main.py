import argparse
import json
import os
import pickle
import sys

import nltk
import numpy as np
import onnxruntime
import pandas as pd

import torch
import torch.onnx
import wandb
import yaml

from config import Config
from src import BaselineQA, FewShotQA_Model
from utils import Trainer, set_seed, Retriever,RetrieverTwoLevel
from data import SQuAD_Dataset, SQuAD_Dataset_fewshot
from nltk.tokenize import sent_tokenize

from onnxruntime.transformers import optimizer as onnx_optimizer
from sentence_transformers import SentenceTransformer
from src import BaselineQA, FewShotQA_Model
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer
from utils import build_tf_idf_wrapper, Retriever, set_seed, store_contents, Trainer
from utils.drqa.DocRanker import docranker_utils
from utils.drqa.DocRanker.tokenizer import CoreNLPTokenizer


def sent_index(text, para, ans_pos):
    if ans_pos == False:
        return -1
    ans = text[2:-2]
    sents = sent_tokenize(para)
    for sent in sents:
        if ans in sent:
            return sents.index(sent)


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data


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

        return (
            con_idx_2_title_idx,
            ques2idx,
            idx2ques,
            con2idx,
            idx2con,
            title2idx,
            idx2title,
        )


def reformat_data_for_sqlite(df, split, use_sentence_level,two_level_drqa=False):
    if (two_level_drqa):
        context_doc_id_original = df[["context", "context_id"]].rename(
            columns={"context": "text", "context_id": "id"})
        context_doc_id_original = context_doc_id_original.drop_duplicates(subset=["text", "id"])
        context_doc_id=pd.DataFrame()
        for idx, row in tqdm(context_doc_id_original.iterrows()):
            context=row['text']
            context_id=row['id']
            # print(context_id,context)
            sent_list=list(sent_tokenize(context))
            sent_id_list= [f"{context_id}_{x}" for x in range(len(sent_list))]
            df_local=pd.DataFrame(list(zip(sent_list,sent_id_list)),columns=['text','id'])
            df_local=df_local.dropna()
            context_doc_id=pd.concat([context_doc_id,df_local],axis=0)
        print(context_doc_id.head())
        context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])	
        context_doc_id.to_json("data-dir/{}/contexts_sentences.json".format(split), orient="records", lines=True)
        context_doc_id = df[["context", "context_id"]].rename(
            columns={"context": "text", "context_id": "id"})
        context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])
        context_doc_id["id"] = context_doc_id["id"].astype(str)
        context_doc_id.to_json("data-dir/{}/contexts_paragraphs.json".format(split), orient="records", lines=True)
    elif (use_sentence_level):
        # TODO: parallelise this using concurrent futures
        context_doc_id_original = df[["context", "context_id"]].rename(
            columns={"context": "text", "context_id": "id"})
        context_doc_id_original = context_doc_id_original.drop_duplicates(subset=["text", "id"])
        context_doc_id=pd.DataFrame()
        for idx, row in tqdm(context_doc_id_original.iterrows()):
            context=row['text']
            context_id=row['id']
            # print(context_id,context)
            sent_list=list(sent_tokenize(context))
            sent_id_list= [f"{context_id}_{x}" for x in range(len(sent_list))]
            df_local=pd.DataFrame(list(zip(sent_list,sent_id_list)),columns=['text','id'])
            df_local=df_local.dropna()
            context_doc_id=pd.concat([context_doc_id,df_local],axis=0)
        print(context_doc_id.head())
        context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])	
    else:
        context_doc_id = df[["context", "context_id"]].rename(
            columns={"context": "text", "context_id": "id"})
        context_doc_id = context_doc_id.drop_duplicates(subset=["text", "id"])
        context_doc_id["id"] = context_doc_id["id"].astype(str)
    context_doc_id.to_json(
        "data-dir/{}/contexts.json".format(split), orient="records", lines=True)


def prepare_retriever(df, db_path, split,use_sentence_level,two_level_drqa=False):
    if (two_level_drqa):
        reformat_data_for_sqlite(df, f"{split}",use_sentence_level,True)
        if (os.path.exists(f"data-dir/{split}/sentence_{db_path}")):
            os.remove(f"data-dir/{split}/sentence_{db_path}")
        
        store_contents(data_path=f"data-dir/{split}/contexts_sentences.json",save_path=f"data-dir/{split}/sentence_{db_path}", preprocess=None, num_workers=2)
        build_tf_idf_wrapper(db_path=f"data-dir/{split}/sentence_{db_path}",
                            out_dir=f"data-dir/{split}", ngram=3, hash_size=(2**25), num_workers=2)

        if (os.path.exists(f"data-dir/{split}/paragraph_{db_path}")):
            os.remove(f"data-dir/{split}/paragraph_{db_path}")
        
        store_contents(data_path=f"data-dir/{split}/contexts_paragraphs.json",save_path=f"data-dir/{split}/paragraph_{db_path}", preprocess=None, num_workers=2)
        build_tf_idf_wrapper(db_path=f"data-dir/{split}/paragraph_{db_path}",
                            out_dir=f"data-dir/{split}", ngram=3, hash_size=(2**25), num_workers=2)	
    else:
        reformat_data_for_sqlite(df, f"{split}",use_sentence_level)
        if (os.path.exists(f"data-dir/{split}/{db_path}")):
            os.remove(f"data-dir/{split}/{db_path}")
        store_contents(data_path=f"data-dir/{split}/contexts.json", save_path=f"data-dir/{split}/{db_path}", preprocess=None, num_workers=2)
        build_tf_idf_wrapper(db_path=f"data-dir/{split}/{db_path}",out_dir=f"data-dir/{split}",ngram=3,hash_size=(2**25),num_workers=2)


def prepare_dense_retriever(tfidf_path, use_sentence_level):
    if "val" in tfidf_path:
        mode = "val"
    else:
        mode = "test"

    assert use_sentence_level, "Dense retriever only works with sentence level"
    assert os.path.exists(tfidf_path), "tfidf_path does not exist"

    _, metadata = docranker_utils.load_sparse_csr(tfidf_path)

    test_data = read_jsonl(f"data-dir/{mode}/contexts.json")

    texts = []
    for id in metadata["doc_dict"][1]:
        for point in test_data:
            if point["id"] == id:
                texts.append(point["text"])

    # check the deivce
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1").to(
        device
    )
    embeddings = (
        model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        .cpu()
        .numpy()
    )

    # save the embeddings
    np.save(
        f"data-dir/{mode}/sentence_transformer_embeddings_multi-qa-distilbert-cos-v1.npy",
        embeddings,
    )


if __name__ == "__main__":
    nltk.download("punkt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Config File")
    parser.add_argument("--top_k", default=10, type=int, help="Topk for retrieval")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
        wandb.init(
            name=config["wandb_path"],
            entity="agv",
            project="interiit-devrev",
            config=config,
        )
        config = Config(**config)

    config.top_k = args.top_k

    set_seed(config.seed)

    # TODO Explore pytorch.quantization also
    assert (not config.quantize or config.ONNX), "Quantizing without ONNX Runtime is not supported"

    print("Reading data csv")
    df_train = pd.read_pickle(config.data.train_data_path)
    df_train=df_train.sample(n=5000)
    df_val = pd.read_pickle(config.data.val_data_path)
    df_test = pd.read_pickle(config.data.test_data_path)

    (
        con_idx_2_title_idx,
        ques2idx,
        idx2ques,
        con2idx,
        idx2con,
        title2idx,
        idx2title,
    ) = load_mappings()

    if config.use_drqa and config.create_drqa_tfidf:
        print("using drqa")
        prepare_retriever(df_val, "sqlite_con.db", "val", config.sentence_level,config.two_level_drqa)
        prepare_retriever(df_train, "sqlite_con.db", "train",False, False)
        prepare_retriever(df_test, "sqlite_con.db", "test", config.sentence_level,config.two_level_drqa)

    if config.create_dense_embeddings:
        print("Creating dense embeddings")
        prepare_dense_retriever(
            "data-dir/val/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz",
            config.sentence_level,
        )
        prepare_dense_retriever(
            "data-dir/test/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz",
            config.sentence_level,
        )

    # add local_files_only=local_files_only if using server
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path,
        TOKENIZERS_PARALLELISM=True,
        model_max_length=512,
        padding="max_length",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.data.apply_aliasing:
        print("Applying aliasing on train data")
        from utils.aliasing import create_alias

        df_train = create_alias(df_train)

    if config.fewshot_qa:
        train_ds = SQuAD_Dataset_fewshot(config, df_train, tokenizer, mask_token)
        val_ds = SQuAD_Dataset_fewshot(config, df_val, tokenizer, mask_token)
        test_ds = SQuAD_Dataset_fewshot(config, df_test, tokenizer, mask_token)

        train_dataloader = DataLoader(
            train_ds,
            batch_size=config.data.train_batch_size,
            collate_fn=train_ds.collate_fn,
        )
        val_dataloader = DataLoader(
            val_ds, batch_size=config.data.val_batch_size, collate_fn=val_ds.collate_fn
        )
        test_dataloader = DataLoader(
            test_ds,
            batch_size=config.data.val_batch_size,
            collate_fn=test_ds.collate_fn,
        )

        model = FewShotQA_Model(config, tokenizer=tokenizer)

        if config.train:
            model.__train__(train_dataloader)

        if config.inference:
            model.__inference__(test_dataloader)

        qa_f1, ttime_per_example = model.few_shot_calculate_metrics(test_dataloader)
        print(f"QA F1: {qa_f1}, Inference time per example: {ttime_per_example} ms")

    else:
        model_verifier=None
        optimizer_verifier=None
        if config.use_verifier:
          config.model.verifier=True
          model_verifier = BaselineQA(config, device).to(device)
          optimizer_verifier = torch.optim.Adam(model_verifier.parameters(), lr=config.training.lr)
          config.model.verifier=False

          if config.load_model_optimizer:
            print(
                "loading verifier model and optimizer from checkpoints/{}/model_optimizer.pt".format(
                    config.verifier_load_path
                )
            )
            checkpoint = torch.load(
                "checkpoints/{}/model_optimizer.pt".format(config.verifier_load_path),
                map_location=torch.device(device),
            )
            model_verifier.load_state_dict(checkpoint["model_state_dict"])
            optimizer_verifier.load_state_dict(checkpoint["optimizer_state_dict"])

        model = BaselineQA(config, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

        if config.load_model_optimizer:
            print(
                "loading model and optimizer from checkpoints/{}/model_optimizer.pt".format(
                    config.load_path
                )
            )
            checkpoint = torch.load(
                "checkpoints/{}/model_optimizer.pt".format(config.load_path),
                map_location=torch.device(device),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


        retriever = None
        if (config.two_level_drqa):
            tfidf_path_para = "data-dir/test/paragraph_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            tfidf_path_sent = "data-dir/test/sentence_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            questions_df = df_test[["question", "title_id"]]
            db_path_sent = "data-dir/test/sentence_sqlite_con.db"
            test_retriever = RetrieverTwoLevel(tfidf_path_sent=tfidf_path_sent, tfidf_path_para=tfidf_path_para, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path_sent=db_path_sent)
      
            tfidf_path_para = "data-dir/val/paragraph_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            tfidf_path_sent = "data-dir/val/sentence_sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            questions_df = df_val[["question", "title_id"]]
            db_path_sent = "data-dir/val/sentence_sqlite_con.db"
            val_retriever = RetrieverTwoLevel(tfidf_path_sent=tfidf_path_sent, tfidf_path_para=tfidf_path_para, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path_sent=db_path_sent)
            
        elif (config.use_drqa):
            tfidf_path = "data-dir/test/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            questions_df = df_test[["question", "title_id"]]
            db_path = "data-dir/test/sqlite_con.db"
            test_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=config.sentence_level,retriever_type=config.retriever_type)
      
            tfidf_path = "data-dir/val/sqlite_con-tfidf-ngram=3-hash=33554432-tokenizer=corenlp.npz"
            questions_df = df_val[["question", "title_id"]]
            db_path = "data-dir/val/sqlite_con.db"
            val_retriever = Retriever(tfidf_path=tfidf_path, questions_df=questions_df, con_idx_2_title_idx=con_idx_2_title_idx, db_path=db_path,sentence_level=config.sentence_level,retriever_type=config.retriever_type)


        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            ques2idx=ques2idx,
            val_retriever=val_retriever,
            df_val=df_val,
            verifier=model_verifier,
            optimizer_verifier=optimizer_verifier
        )

        if config.train:
            print("Creating train dataset")
            train_ds = SQuAD_Dataset(config, df_train, tokenizer)
            train_dataloader = DataLoader(
                train_ds,
                batch_size=config.data.train_batch_size,
                collate_fn=train_ds.collate_fn,
            )
            print("length of train dataset: {}".format(train_ds.__len__()))

            print("Creating val dataset")
            val_ds = SQuAD_Dataset(config, df_val, tokenizer)
            val_dataloader = DataLoader(
                val_ds,
                batch_size=config.data.val_batch_size,
                collate_fn=val_ds.collate_fn,
            )
            print("length of val dataset: {}".format(val_ds.__len__()))

            trainer.train(train_dataloader, val_dataloader)

    if config.inference:
        # print("Creating test dataset")
        # test_ds = SQuAD_Dataset(config, df_test, tokenizer)
        # test_dataloader = DataLoader(test_ds, batch_size=config.data.val_batch_size, collate_fn=test_ds.collate_fn)
        # print("length of test dataset: {}".format(test_ds.__len__()))

        # calculate_metrics(test_ds, test_dataloader, wandb_logger)
        # test_metrics = trainer.calculate_metrics(test_ds, test_dataloader)
        if config.train and config.save_model_optimizer:
            print(
                "loading best model from checkpoints/{}/model_optimizer.pt for inference".format(
                    config.load_path
                )
            )
            checkpoint = torch.load(
                "checkpoints/{}/model_optimizer.pt".format(config.load_path),
                map_location=torch.device(device),
            )
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            if config.use_verifier:
              print(
                  "loading best verifier model from checkpoints/{}/model_optimizer.pt for inference".format(
                      config.verifier_load_path
                  )
              )
              checkpoint = torch.load(
                  "checkpoints/{}/model_optimizer.pt".format(config.verifier_load_path),
                  map_location=torch.device(device),
              )
              trainer.verifier.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.inference_device)
        test_metrics = trainer.calculate_metrics(
            df_test, test_retriever, "test", config.inference_device, do_prepare=True
        )
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
