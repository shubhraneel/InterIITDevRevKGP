import time
import sklearn
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import compute_f1
from data import SQuAD_Dataset

class Trainer():

    def __init__(self, config, model, optimizer, device):
        self.config = config
        self.model = model
        self.device = device

        self.optimizer = optimizer
    

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
            tepoch.set_postfix(loss = total_loss / (batch_idx+1))
            
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / batch_idx


    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(train_dataloader, epoch)
            
            if ((val_dataloader is not None) and (((epoch + 1) % self.config.training.evaluate_every)) == 0):
                self.evaluate(val_dataloader)

    def evaluate(self, dataloader):
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

        return total_loss / batch_idx

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

        for batch_idx, batch in enumerate(tepoch):
            
            # list of titles in the batch 
            title = batch["title"]  

            # list of paragraph indices (in dataset.data) for each question in the batch
            para_ids_batch = [dataset.theme_para_id_mapping[t] for t in title]    # List of para ids for each question in the batch

            # iterate over questions in the batch
            for question_idx in range(len(para_ids_batch)):
                question = batch["question"][question_idx]
                
                # list of paragraph ids in the same theme as q
                q_para_ids = para_ids_batch[question_idx]

                # list of paragraphs for in the same theme as q
                paragraphs = list(set(dataset.df.iloc[q_para_ids]["Paragraph"]))

                # question id (primary key in df)
                q_id = batch["id"][question_idx]

                # create kb dataframe
                df_kb = self._create_inference_df(question, dataset, paragraphs, q_id)
                # print(q_para_ids)
                # print(len(df_kb))

                temp_ds = SQuAD_Dataset(dataset.config, df_kb, dataset.tokenizer)
                temp_dataloader = DataLoader(temp_ds, batch_size=dataset.config.data.val_batch_size, collate_fn=temp_ds.collate_fn)

                total_time_per_question = 0
                
                # loop for iterating over question para pairs to extract paras
                for qp_batch_id, qp_batch in enumerate(temp_dataloader):
                    start_time = time.time()
                    if (len(qp_batch["question_context_input_ids"].shape) == 1):
                        qp_batch["question_context_input_ids"] = qp_batch["question_context_input_ids"].unsqueeze(dim=0)
                        qp_batch["question_context_attention_mask"] = qp_batch["question_context_attention_mask"].unsqueeze(dim=0)
                        if not self.config.model.non_pooler:
                            qp_batch["question_context_token_type_ids"] = qp_batch["question_context_token_type_ids"].unsqueeze(dim=0)
                    
                    pred = self.predict(qp_batch)

                    offset_mappings_list = qp_batch["question_context_offset_mapping"]
                    contexts_list = qp_batch["context"]

                    gold_answers.extend([answer[0] if (len(answer) != 0) else "" for answer in qp_batch["answer"]])

                    pred_start_index = torch.argmax(pred.start_logits, axis=1)
                    pred_end_index = torch.argmax(pred.end_logits, axis=1)

                    # iterate over each context
                    for c_id, context in enumerate(contexts_list):
                        # TODO: don't take only best pair (see HF tutorial)

                        pred_answer = ""
                        if (offset_mappings_list[c_id][pred_start_index[c_id]] is not None and offset_mappings_list[c_id][pred_end_index[c_id]]):
                            try:
                                pred_start_char = offset_mappings_list[c_id][pred_start_index[c_id]][0]
                                pred_end_char = offset_mappings_list[c_id][pred_end_index[c_id]][1]
                            
                            except:
                                print(offset_mappings_list[c_id])
                                raise ValueError

                            pred_answer = context[pred_start_char:pred_end_char]

                        predicted_answers.append(pred_answer)
                        
                    # TODO: remove offset_mapping etc. lookup from inference time (current calculation is the absolute worst case time)
                    total_time_per_question += (time.time() - start_time)
                    total_time_per_question_list.append(total_time_per_question)

        results = {
                    "mean_time_per_question": np.mean(np.array(total_time_per_question_list)),
                    "predicted_answers": predicted_answers,
                    "gold_answers": gold_answers,
                }     

        return results

    def calculate_metrics(self, dataset, dataloader, logger=None):
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
        results = self.inference(dataset, dataloader)
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

        metrics = {
            "classification_f1": classification_f1,
            "mean_squad_f1": mean_squad_f1,
            "mean_time_per_question (ms)": results["mean_time_per_question"]*1000,
        }

        return metrics
