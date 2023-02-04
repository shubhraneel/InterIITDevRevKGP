import argparse
import json
import os
import pickle
import sys
import random
import gc 

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

from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import FAISSDocumentStore

from collections import defaultdict
from sklearn import metrics
from time import time
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time 
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing 

def fit_and_evaluate( km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    for seed in range(n_runs):
        km.set_params(random_state=seed)
        km.fit(X)

    return km

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

    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1").to(
        device
    )
    embeddings = (
        model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
        .cpu()
        .numpy()
    )

    # save the embeddings
    np.save(
        f"data-dir/{mode}/sentence_transformer_embeddings_multi-qa-mpnet-base-dot-v1.npy",
        embeddings,
    )


if __name__ == "__main__":
    nltk.download("punkt")
    import logging

    # logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Config File")
    parser.add_argument("--top_k", default=None, type=int, help="Topk for retrieval")

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
    
    if args.top_k is not None:
        config.top_k = args.top_k

    set_seed(config.seed)

    # TODO Explore pytorch.quantization also
    assert (not config.quantize or config.ONNX), "Quantizing without ONNX Runtime is not supported"

    print("Reading data csv")
    df_train = pd.read_pickle(config.data.train_data_path)
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

    if not config.use_dpr and config.create_dense_embeddings:
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

    
   
    model = BaselineQA(config, device).to(device)

    if config.load_model_optimizer:
        print(
            "loading model and optimizer from checkpoints/{}/model_optimizer.pt".format(
                config.load_path
            )
        )
        checkpoint = torch.load(
            "checkpoints/{}/avg_model.pt".format(config.load_path),
            map_location=torch.device(device),
        )
        model.load_state_dict(checkpoint["model_state_dict"])


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
    elif config.use_dpr:
        query_model = config.retriever.query_model
        passage_model = config.retriever.passage_model

        unique_data = df_test.drop_duplicates(subset='context', keep="first")
        UniqueParaList = unique_data.context.to_list()
        ThemeList = unique_data.title.to_list()
        if not os.path.exists("data-dir/test_paragraphs/"):
            os.mkdir("data-dir/test_paragraphs/")
        for i in range(len(UniqueParaList)):
            with open("data-dir/test_paragraphs/Paragraph_" + str(i) + ".txt", 'w+') as fp:
                fp.write("%s\n" % UniqueParaList[i])
        if config.create_dense_embeddings:
            if (os.path.exists("data-dir/faiss_document_store_test.db")):
                os.remove("data-dir/faiss_document_store_test.db")
            test_document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", sql_url="sqlite:///data-dir/faiss_document_store_test.db")
            test_docs = convert_files_to_docs(dir_path="data-dir/test_paragraphs/", clean_func=clean_wiki_text, split_paragraphs=True)
            test_document_store.write_documents(test_docs)
            test_retriever = DensePassageRetriever(
                document_store=test_document_store,
                query_embedding_model=query_model,
                passage_embedding_model=passage_model,
                max_seq_len_query=64,
                max_seq_len_passage=512,
            )
            test_document_store.update_embeddings(test_retriever)
        #     with open("data-dir/test_retriever.pkl","wb") as f:
        #         pickle.dump(test_retriever, f)
        #     with open("data-dir/test_document_store.pkl","wb") as f:
        #         pickle.dump(test_document_store, f)
        # else:
        #     with open("data-dir/test_retriever.pkl","rb") as f:
        #         test_retriever = pickle.load(f)
        #     with open("data-dir/test_document_store.pkl","rb") as f:
        #         test_document_store = pickle.load(f)

        unique_data = df_val.drop_duplicates(subset='context', keep="first")
        UniqueParaList = unique_data.context.to_list()
        ThemeList = unique_data.title.to_list()
        if not os.path.exists("data-dir/val_paragraphs/"):
            os.mkdir("data-dir/val_paragraphs/")
        for i in range(len(UniqueParaList)):
            with open("data-dir/val_paragraphs/Paragraph_" + str(i) + ".txt", 'w+') as fp:
                fp.write("%s\n" % UniqueParaList[i])
        if config.create_dense_embeddings:
            if (os.path.exists("data-dir/faiss_document_store_val.db")):
                os.remove("data-dir/faiss_document_store_val.db")
            val_document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", sql_url="sqlite:///data-dir/faiss_document_store_val.db")
            val_docs = convert_files_to_docs(dir_path="data-dir/val_paragraphs/", clean_func=clean_wiki_text, split_paragraphs=True)
            val_document_store.write_documents(val_docs)
            val_retriever = DensePassageRetriever(
                document_store=val_document_store,
                query_embedding_model=query_model,
                passage_embedding_model=passage_model,
                max_seq_len_query=64,
                max_seq_len_passage=512,
            )
            val_document_store.update_embeddings(val_retriever)



        

        

    if config.save_model_optimizer:
        print(
            "saving model and optimizer at checkpoints/{}/model_optimizer.pt".format(
                config.load_path
            )
        )
        os.makedirs("checkpoints/{}/".format(config.load_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict()
            },
            "checkpoints/{}/avg_model.pt".format(config.load_path),
        )

    if config.federated.use and config.train:

        # def add_dummy_cluster_id(df_train):
        #     dict_cluster_id = {
        #         title_id: random.randint(0, config.federated.num_clusters-1) 
        #         for title_id in df_train['title_id'].unique()
        #     }
        #     cluster_ids = [dict_cluster_id[title_id] for title_id in df_train['title_id']]
        #     df_train['cluster_id'] = cluster_ids
        #     return df_train

        def add_cluster_id(df_train):
            str_list=[]
            theme_list=[]

            df=df_train
            df=df[['title','context']]
            df=df.drop_duplicates(subset="context")

            for theme in df.title.unique():
                new_df=df.loc[df['title']==theme]
                str=""
                
                for i in range(new_df.shape[0]):
                    str+=new_df.iloc[i].context
                
                str_list.append(str)
                theme_list.append(theme)

            if config.load_model_optimizer:
                with open(f"checkpoints/{config.federated.cluster_path}/tfidf_model.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                with open(f"checkpoints/{config.federated.cluster_path}/lsa_model.pkl", "rb") as f:
                    lsa = pickle.load(f)
                with open(f"checkpoints/{config.federated.cluster_path}/kmeans_model.pkl", "rb") as f:
                    kmeans = pickle.load(f)
                
                X_tfidf = vectorizer.transform(str_list)
                X_lsa = lsa.transform(X_tfidf)
                labels = kmeans.predict(X_lsa)

            else:
                vectorizer = TfidfVectorizer()
                X_tfidf = vectorizer.fit_transform(str_list)

                lsa = make_pipeline(TruncatedSVD(n_components=config.federated.lsa_dim_reduction), Normalizer(copy=False))
                X_lsa = lsa.fit_transform(X_tfidf)

                kmeans = KMeans(
                                n_clusters=config.federated.num_clusters,
                                max_iter=100,
                                n_init=5,
                                random_state=4
                                )
                

                # def fit_and_evaluate( km, X, labels, evaluations =[] , evaluations_std = [], name=None, n_runs=5)
                # le = preprocessing.LabelEncoder()
                # labels=le.fit_transform(df_train.title)
                                
                kmeans= fit_and_evaluate(
                                kmeans,
                                X_lsa,
                                name="KMeans\nwith LSA on tf-idf vectors",
                                )
                
                labels = kmeans.labels_
            


            if config.save_model_optimizer:
                with open(f"checkpoints/{config.federated.cluster_path}/tfidf_model.pkl", "wb+") as f:
                    pickle.dump(vectorizer, f)
                with open(f"checkpoints/{config.federated.cluster_path}/svdnorm_model.pkl", "wb+") as f:
                    pickle.dump(lsa, f)
                with open(f"checkpoints/{config.federated.cluster_path}/kmeans_model.pkl", "wb+") as f:
                    pickle.dump(kmeans, f)

            dictionary = {k: v for k, v in zip(theme_list, labels)}
            cluster_ids = [dictionary[title] for title in df_train['title']]

            df_train['cluster_id'] = cluster_ids

            return df_train

        num_rounds=config.training.epochs//config.federated.num_epochs
        print("Creating train dataset")
        train_dataloader_dict = {i:None  for i in range(config.federated.num_clusters)}
        df_train=add_cluster_id(df_train=df_train)

        for i in range(config.federated.num_clusters):
            train_ds = SQuAD_Dataset(config, df_train[df_train['cluster_id']==i], tokenizer)
            train_dataloader = DataLoader(
                train_ds,
                batch_size=config.data.train_batch_size,
                collate_fn=train_ds.collate_fn,
            )
            train_dataloader_dict[i]=train_dataloader
            print("length of train dataset: {}".format(train_ds.__len__()))

        print("Creating val dataset")
        val_ds = SQuAD_Dataset(config, df_val, tokenizer)
        val_dataloader = DataLoader(
            val_ds,
            batch_size=config.data.val_batch_size,
            collate_fn=val_ds.collate_fn,
        )
        print("length of val dataset: {}".format(val_ds.__len__()))
        
        config.training.epochs = config.federated.num_epochs
        
        for round in range(num_rounds):
            
            for i in range(config.federated.num_clusters):
                model = BaselineQA(config, device).to(device)
                checkpoint = torch.load(
                    "checkpoints/{}/avg_model.pt".format(config.load_path),
                    map_location=torch.device(device),
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(config.inference_device)

                trainer = Trainer(
                    config=config,
                    model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=config.training.lr),
                    device=device,
                    tokenizer=tokenizer,
                    ques2idx=ques2idx,
                    val_retriever=val_retriever,
                    df_val=df_val,
                )
                print(f" runninng round {round} on cluster {i}")
                trainer.train(train_dataloader_dict[i], val_dataloader)
                
                os.makedirs(f"checkpoints/{config.load_path}/{i}/", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict()
                    },
                    "checkpoints/{}/{}/model.pt".format(config.load_path,i),
                )
                del model
                del trainer
                del checkpoint
                gc.collect()                
            
            model = BaselineQA(config,device).to(device)
            state_dict = model.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = torch.zeros_like(state_dict[key])
            for i in range(config.federated.num_clusters):
                checkpoint = torch.load(
                    "checkpoints/{}/{}/model.pt".format(config.load_path,i),
                    map_location=torch.device(device),
                )
                client_state_dict = checkpoint['model_state_dict']
                for key, param in state_dict.items():
                    state_dict[key] += client_state_dict[key]
                del checkpoint
                del client_state_dict
                gc.collect()

            for key, param in state_dict.items():
                if (state_dict[key].dtype == torch.int64):
                    state_dict[key] = torch.div(state_dict[key],config.federated.num_clusters,rounding_mode='floor').to(torch.int64)
                else:
                    state_dict[key] *= (1/config.federated.num_clusters) 

            model.load_state_dict(state_dict)
            trainer = Trainer(
                    config=config,
                    model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=config.training.lr),
                    device=device,
                    tokenizer=tokenizer,
                    ques2idx=ques2idx,
                    val_retriever=val_retriever,
                    df_val=df_val,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict()
                },
                "checkpoints/{}/avg_model.pt".format(config.load_path),
            )
            print(f"val metrics of avg model at round {round}")
            val_metrics = trainer.calculate_metrics(
                df_val, val_retriever, "val", config.inference_device, do_prepare=True
            )
            print(val_metrics)
            del model
            del state_dict
            gc.collect()
        

    if config.inference:
        model = BaselineQA(config, device).to(device)
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
            model.load_state_dict(checkpoint["model_state_dict"])
        
        trainer = Trainer(
                    config=config,
                    model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=config.training.lr),
                    device=device,
                    tokenizer=tokenizer,
                    ques2idx=ques2idx,
                    val_retriever=val_retriever,
                    df_val=df_val,
        )
        checkpoint = torch.load(
                    "checkpoints/{}/avg_model.pt".format(config.load_path),
                    map_location=torch.device(device),
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.inference_device)
        test_metrics = trainer.calculate_metrics(
            df_test, test_retriever, "test", config.inference_device, do_prepare=True
        )
        print(test_metrics)
