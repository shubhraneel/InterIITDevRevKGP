import time
import wandb
import sklearn
import numpy as np 
import pandas as pd
from tqdm import tqdm

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import compute_f1
from data import SQuAD_Dataset

class Trainer():
    def __init__(self, config, model, optimizer, device, tokenizer, retriever=None):
        self.config = config
        self.device = device

        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.model = model

        wandb.watch(self.model)

        self.retriever = retriever

    def _train_step(self, dataloader, epoch):
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")
            if (len(batch["question_context_input_ids"].shape) == 1):
                batch["question_context_input_ids"] = batch["question_context_input_ids"].unsqueeze(dim=0)
                batch["question_context_attention_mask"] = batch["question_context_attention_mask"].unsqueeze(dim=0)
                if not self.config.model.non_pooler:
                    batch["question_context_token_type_ids"] = batch["question_context_token_type_ids"].unsqueeze(dim=0)

            out = self.model(batch)
            loss = out.loss
            loss.backward()

            total_loss += loss.item()
            tepoch.set_postfix(loss = total_loss / (batch_idx + 1))
            wandb.log({"train_batch_loss": total_loss / (batch_idx + 1)})
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        wandb.log({"train_epoch_loss": total_loss / (batch_idx + 1)})

        return (total_loss / (batch_idx + 1))


    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(train_dataloader, epoch)
            
            if ((val_dataloader is not None) and (((epoch + 1) % self.config.training.evaluate_every)) == 0):
                self.evaluate(val_dataloader)
                self.model.train()


    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Validation Step")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tepoch):
                if (len(batch["question_context_input_ids"].shape) == 1):
                    batch["question_context_input_ids"] = batch["question_context_input_ids"].unsqueeze(dim=0)
                    batch["question_context_attention_mask"] = batch["question_context_attention_mask"].unsqueeze(dim=0)
                    if not self.config.model.non_pooler:
                        batch["question_context_token_type_ids"] = batch["question_context_token_type_ids"].unsqueeze(dim=0)
                    
                out = self.model(batch)
                loss = out.loss

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))
                wandb.log({"val_batch_loss": total_loss / (batch_idx + 1)})

        wandb.log({"val_epoch_loss": total_loss / (batch_idx + 1)})
        return (total_loss / (batch_idx + 1))


    def _create_inference_df(self, question, dataset, paragraphs, id):
        """
        Create a dataframe with all paragraphs corresponding to a particular question
        """
        
        # TODO: Optimize by replacing loc with query function 
        df_kb = pd.DataFrame()
        for p in paragraphs:
            # if (q, p) is already in the df, just return the answer
            df_in_data = dataset.df.loc[(dataset.df["Question"] == question) & (dataset.df["Paragraph"] == p)]
            if (len(df_in_data) != 0):
                try:
                    assert len(df_in_data) == 1
                    df_kb = pd.concat([df_kb, df_in_data], axis = 0).reset_index(drop = True)
                except:
                    print(len(df_in_data))
                    print(type(df_in_data))
                    print(df_in_data.to_dict())
                    raise 
                # assert len(df_in_data) == 1
                # df_kb = pd.concat([df_kb, df_in_data], axis = 0).reset_index(drop = True)
            else:
                # Unnamed: 0	Theme	Paragraph	Question	Answer_possible	Answer_text	Answer
                # kb_dict = {"Unnamed: 0": id, "Question": question, ""}
                kb_row = dataset.df.loc[dataset.df["Unnamed: 0"] == id].to_dict(orient = "records")[0]
                kb_row["Paragraph"] = p
                kb_row["Answer_possible"] = False
                kb_row["Answer_start"] = []
                kb_row["Answer_text"] = []

                # df_in_data = pd.DataFrame(kb_row, index = False)

                df_kb = pd.concat([df_kb, pd.DataFrame(kb_row)], axis=0)

                # df_kb = pd.concat([df_kb, df_in_data], axis = 0).reset_index(drop = True)
                # kb_row["Question"] = question
                # return unanswerable (Answerable_possible = False, Answer_text = [], Answer_start = [])

        return df_kb

    def predict(self, batch):
        return self.model(batch)

    def inference(self, dataset, dataloader):
        # TODO: use only dataset (applying transforms as done in collate_fn here itself)
        self.model.to(self.config.inference_device)
        self.device = self.config.inference_device
        self.model.device = self.config.inference_device

        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        tepoch.set_description("Inference Step")

        total_time_per_question_list = []
        predicted_answers = []
        gold_answers = []

        # TODO: is this time calculation correct?
        for batch_idx, batch in enumerate(tepoch):
            
            # list of titles in the batch 
            title_list = batch["title"]  

            # list of paragraph indices (in dataset.data) for each question in the batch
            para_ids_batch = [dataset.theme_para_id_mapping[t] for t in title_list]

            # iterate over questions in the batch
            for question_idx in range(len(batch["question"])):
                start_time = time.time()

                question = batch["question"][question_idx]
                theme_id = str(batch["theme_id"][question_idx])
                
                # create knowledge base dataframe containing all paragraphs of the same theme as q
                if (self.config.use_drqa):
                    doc_names_filtered, doc_text_filtered = self.retriever.retrieve_top_k(question, theme_id, k=self.config.drqa_top_k)

                    df_kb = pd.DataFrame()
                    df_kb["Paragraph"] = doc_text_filtered
                    df_kb["paragraph_id"] = doc_names_filtered

                    df_kb["Question"] = question
                    df_kb["question_id"] = batch["question_id"][question_idx]

                    df_kb["Theme"] = batch["title"][question_idx]
                    df_kb["theme_id"] = batch["theme_id"][question_idx]

                    df_kb["Answer_possible"] = False
                    df_kb["Answer_start"] = "[]"#*len(doc_names_filtered)
                    df_kb["Answer_text"] = "[]"#*len(doc_names_filtered)    

                    row_in_data = dataset.df.loc[dataset.df["question_id"] == batch["question_id"][question_idx]]

                    # print(df_kb["paragraph_id"])
                    # print(row_in_data["paragraph_id"])

                    try:
                        assert (len(row_in_data) == 1)
                    except:
                        print(row_in_data)
                        raise  

                    # TODO: Optimize
                    row_in_data_idx = df_kb.loc[df_kb["paragraph_id"].astype(str) == str(row_in_data["paragraph_id"].values[0])].index
                    # print(row_in_data, row_in_data_idx)
                    # print("row_in_data['paragraph_id'].values[0]", row_in_data["paragraph_id"].values[0])
                    df_kb.loc[row_in_data_idx, "Answer_possible"] = row_in_data["Answer_possible"].values[0]
                    df_kb.loc[row_in_data_idx, "Answer_start"] = row_in_data["Answer_start"].values[0]
                    df_kb.loc[row_in_data_idx, "Answer_text"] = row_in_data["Answer_text"].values[0]

                else:
                    # list of paragraph ids in the same theme as q
                    q_para_ids = para_ids_batch[question_idx]

                    # list of paragraphs for in the same theme as q
                    paragraphs = list(set(dataset.df.iloc[q_para_ids]["Paragraph"]))

                    # question id (primary key in df)
                    q_id = batch["question_id"][question_idx]

                    df_kb = self._create_inference_df(question, dataset, paragraphs, q_id)

                # TODO: keep more than 1 question per dataloader for max util (keep stride = k in line 143)
                temp_ds = SQuAD_Dataset(dataset.config, df_kb, dataset.tokenizer, hide_tqdm=True)
                temp_dataloader = DataLoader(temp_ds, batch_size=dataset.config.data.val_batch_size, collate_fn=temp_ds.collate_fn)

                # print(question)
                # print(len(temp_ds))
                
                # loop for iterating over question para pairs to extract paras
                best_prob = 0
                best_row = None
                for qp_batch_id, qp_batch in enumerate(temp_dataloader):
                    if (len(qp_batch["question_context_input_ids"].shape) == 1):
                        qp_batch["question_context_input_ids"] = qp_batch["question_context_input_ids"].unsqueeze(dim=0)
                        qp_batch["question_context_attention_mask"] = qp_batch["question_context_attention_mask"].unsqueeze(dim=0)
                        if not self.config.model.non_pooler:
                            qp_batch["question_context_token_type_ids"] = qp_batch["question_context_token_type_ids"].unsqueeze(dim=0)
                    
                    pred = self.predict(qp_batch)

                    # offset_mappings_list = qp_batch["question_context_offset_mapping"]
                    # contexts_list = qp_batch["context"]

                    # gold_answers.extend([answer[0] if (len(answer) != 0) else "" for answer in qp_batch["answer"]])

                    pred_start = torch.max(pred.start_logits, axis=1)
                    pred_start_index = pred_start.indices
                    pred_start_prob = pred_start.values

                    pred_end = torch.max(pred.end_logits, axis=1)
                    pred_end_index = pred_end.indices
                    pred_end_prob = pred_end.values

                    # to predict only one answer
                    batch_best = torch.max(pred_start_prob*pred_end_prob, axis=0)
                    batch_best_paragraph_idx = batch_best.indices.item()
                    batch_best_paragraph_prob = batch_best.values.item()

                    # print("batch_best_paragraph_idx", batch_best_paragraph_idx)
                    # print("batch_best_paragraph_prob", batch_best_paragraph_prob)

                    if (batch_best_paragraph_prob > best_prob):
                        best_prob = batch_best_paragraph_prob
                        best_row = {key: qp_batch[key][batch_best_paragraph_idx] for key in ["context", "answer", "question_context_offset_mapping"]}
                        best_row["pred_start_index"] = pred_start_index[batch_best_paragraph_idx]
                        best_row["pred_end_index"] = pred_end_index[batch_best_paragraph_idx]

                    # to predict for all paragraphs
                    # # iterate over each context
                    # for c_id, context in enumerate(contexts_list):
                    #     # TODO: don't take only best pair (see HF tutorial)

                    #     pred_answer = ""
                    #     if (offset_mappings_list[c_id][pred_start_index[c_id]] is not None and offset_mappings_list[c_id][pred_end_index[c_id]] is not None):
                    #         try:
                    #             pred_start_char = offset_mappings_list[c_id][pred_start_index[c_id]][0]
                    #             pred_end_char = offset_mappings_list[c_id][pred_end_index[c_id]][1]
                    #         except:
                    #             print(offset_mappings_list[c_id])
                    #             raise ValueError

                    #         pred_answer = context[pred_start_char:pred_end_char]

                    #     predicted_answers.append(pred_answer)
                        
                    # TODO: remove offset_mapping etc. lookup from inference time (current calculation is the absolute worst case time)

                total_time_per_question = (time.time() - start_time)
                total_time_per_question_list.append(total_time_per_question)
                wandb.log({"inference_time_per_question": total_time_per_question})

                try:
                    gold_answers.append(best_row["answer"][0] if (len(best_row["answer"]) != 0) else "") 
                except:
                    print("best_row", best_row)
                    print("df_kb", df_kb)
                    print("doc_text_filtered", doc_text_filtered)
                    print("question", question)
                    print("theme_id", theme_id)
                    print("row_in_data", row_in_data)
                    raise

                offset_mappings = best_row["question_context_offset_mapping"]
                context = best_row["context"]

                pred_answer = ""
                if (offset_mappings[best_row["pred_start_index"]] is not None and offset_mappings[best_row["pred_end_index"]] is not None):
                    pred_start_char = offset_mappings[best_row["pred_start_index"]][0]
                    pred_end_char = offset_mappings[best_row["pred_end_index"]][1]

                    pred_answer = context[pred_start_char:pred_end_char]
                
                predicted_answers.append(pred_answer)

        results = {
                    "mean_time_per_question": np.mean(np.array(total_time_per_question_list)),
                    "predicted_answers": predicted_answers,
                    "gold_answers": gold_answers,
                }     

        return results, predicted_answers, gold_answers

    def calculate_metrics(self, dataset, dataloader):
        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        # TODO: check if this way of calculating the time is correct
        torch.cuda.synchronize()
        # tsince = int(round(time.time() * 1000))
        # results = self.__inference__(dataset, dataloader, logger)
        results, predicted_answers, gold_answers = self.inference(dataset, dataloader)
        torch.cuda.synchronize()
        # ttime_elapsed = int(round(time.time() * 1000)) - tsince
        # print ('test time elapsed {}ms'.format(ttime_elapsed))

        # ttime_per_example = (ttime_elapsed * dataloader.batch_size)/len(results["ground"])
        # classification_f1 = sklearn.metrics.f1_score(results["preds"], results["ground"]) # For paragraph search task

        # TODO/DOUBT: should we add a classification filter first? 

        squad_f1_per_span = []
        for i in range(len(results["predicted_answers"])):
            squad_f1_per_span.append(compute_f1(results["predicted_answers"][i], results["gold_answers"][i])) # For the text
        mean_squad_f1 = np.mean(squad_f1_per_span)

        classification_prediction = [1 if (len(results["predicted_answers"][i]) != 0) else 0 for i in range(len(results["predicted_answers"])) ] 
        classification_actual = [1 if (len(results["gold_answers"][i]) != 0) else 0 for i in range(len(results["gold_answers"])) ] 
        classification_f1 = sklearn.metrics.f1_score(classification_actual, classification_prediction)
        classification_accuracy = sklearn.metrics.accuracy_score(classification_actual, classification_prediction)

        metrics = {
            "classification_accuracy": classification_accuracy,
            "classification_f1": classification_f1,
            "mean_squad_f1": mean_squad_f1,
            "mean_time_per_question (ms)": results["mean_time_per_question"]*1000,
        }

        wandb.log({"metrics": metrics})
        wandb.log({"predicted_answers": predicted_answers})
        wandb.log({"gold_answers": gold_answers})

        return metrics