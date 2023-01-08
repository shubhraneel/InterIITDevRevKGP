import time
import json
import wandb
import numpy as np 
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import compute_f1
from data import SQuAD_Dataset, BiDAF_Dataset

from data import *

class Trainer():
    def __init__(self, config, model, optimizer, device, tokenizer, ques2idx, retriever=None):
        self.config = config
        self.device = device

        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.model = model

        wandb.watch(self.model)

        self.ques2idx = ques2idx

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
            if self.config.model.two_step_loss:
                out=out[0]
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
                if self.config.model.two_step_loss:
                    out=out[0]
                loss = out.loss

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))
                wandb.log({"val_batch_loss": total_loss / (batch_idx + 1)})

        wandb.log({"val_epoch_loss": total_loss / (batch_idx + 1)})
        return (total_loss / (batch_idx + 1))


    def predict(self, batch):
        return self.model(batch)


    def inference(self, df_test):
        self.model.to(self.config.inference_device)
        self.device = self.config.inference_device
        self.model.device = self.config.inference_device

        title_id_list = df_test["title_id"].unique()
        gb_title = df_test.groupby("title_id")
        df_test_matched = pd.DataFrame()

        df_unique_con = df_test.drop_duplicates(subset=["context"])
        unmatched=0

        start_time = time.time()
        for title_id in title_id_list:
            df_temp = gb_title.get_group(title_id)
            
            for idx, row in df_temp.iterrows():
                question = row["question"]
                question_id = row["question_id"]
                context_id = row["context_id"]

                df_contexts=pd.DataFrame()
                if self.retriever is not None:
                    doc_idx_filtered, doc_text_filtered = self.retriever.retrieve_top_k(question, str(title_id), k=self.config.top_k)
                    df_contexts_og = df_unique_con.loc[df_unique_con["context_id"].isin([int(doc_idx) for doc_idx in doc_idx_filtered])].copy()
                    # TODO: we can endup sampling things in doc_idx_filtered again
                    df_contexts_random =  df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=max(0,self.config.top_k-len(doc_idx_filtered)),random_state=self.config.seed)
                    df_contexts = pd.concat([df_contexts_og, df_contexts_random], axis=0, ignore_index=True)
                else:
                    df_contexts = df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=self.config.top_k,random_state=self.config.seed)
                df_contexts.loc[:, "question"] = question
                df_contexts.loc[:, "question_id"] = question_id
                df_contexts.loc[:, "answerable"] = False
                df_contexts.loc[:, "answer_start"] = ""
                df_contexts.loc[:, "answer_text"] = ""
                
                original_context_idx = df_contexts.loc[df_contexts["context_id"] == context_id]
                if (len(original_context_idx) == 0):    
                    # print(f"original paragraph not in top k {unmatched}")
                    unmatched+=1
                    # don't uncomment this
                    # df_contexts = pd.concat([df_contexts, pd.DataFrame(row)], axis=0, ignore_index=True)
                else:
                    row_dict = row.to_dict()
                    df_contexts.loc[df_contexts["context_id"] == context_id, row_dict.keys()] = row_dict.values()
                df_test_matched = pd.concat([df_test_matched, df_contexts], axis=0, ignore_index=True)

        # print(f"original paragraph not in top k {unmatched}")
        
        test_ds = SQuAD_Dataset(self.config, df_test_matched, self.tokenizer) #, hide_tqdm=True
        test_dataloader = DataLoader(test_ds, batch_size=self.config.data.val_batch_size, collate_fn=test_ds.collate_fn)
        time_test_dataloader_generation=1000*(time.time() - start_time)
        print(time_test_dataloader_generation)
        print(time_test_dataloader_generation/df_test.shape[0])
        wandb.log({"time_test_dataloader_generation": time_test_dataloader_generation})
        wandb.log({"per_q_time_test_dataloader_generation": time_test_dataloader_generation/df_test.shape[0]})
        
        print(f"{len(df_test_matched)=}")
        print(f"{len(df_temp)=}")
       
        start_time=time.time()
        question_prediction_dict={q_id:(0,"") for q_id in df_test_matched["question_id"].unique()}

        # TODO: without sequentional batch iteration
        for qp_batch_id, qp_batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            
            if (len(qp_batch['question_context_input_ids'].shape) == 1):
                qp_batch['question_context_input_ids'] = qp_batch['question_context_input_ids'].unsqueeze(dim=0)
                qp_batch['question_context_attention_mask'] = qp_batch['question_context_attention_mask'].unsqueeze(dim=0)
                if not self.config.model.non_pooler:
                    qp_batch['question_context_token_type_ids'] = qp_batch['question_context_token_type_ids'].unsqueeze(dim=0)
            
            # para, para_id, theme, theme_id, question, question_id
            pred = self.predict(qp_batch)
            if self.config.model.two_step_loss:
                confidence_scores=pred[1] # -> [32,1]
                # pred=pred[0]
            else:
                # print(pred.start_logits.shape) # -> [32,512] 
                start_probs=F.softmax(pred.start_logits,dim=1)  # -> [32,512] 
                end_probs=F.softmax(pred.end_logits,dim=1)    # -> [32,512] 

                max_start_probs=torch.max(start_probs, axis=1)  # -> [32,1] 
                max_end_probs=torch.max(end_probs,axis=1)       # -> [32,1]

                confidence_scores=max_end_probs.values*max_start_probs.values  # -> [32,1]                

            for batch_idx,q_id in enumerate(qp_batch["question_id"]):
                if (question_prediction_dict[q_id][0]<confidence_scores[batch_idx]):
                    # using the context in the qp_pair get extract the span using max_start_prob and max_end_prob                    
                    context = qp_batch["context"][batch_idx]
                    offset_mapping = qp_batch["question_context_offset_mapping"][batch_idx]
                    decoded_answer = ""
                    
                    start_index = max_start_probs.indices[batch_idx].item()
                    end_index = max_end_probs.indices[batch_idx].item()

                    if (offset_mapping[start_index] is not None and offset_mapping[end_index] is not None):
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        decoded_answer = context[start_char:end_char]
                    question_prediction_dict[q_id]=(confidence_scores[batch_idx].item(),decoded_answer)
    
        time_inference_generation=1000*(time.time()-start_time)
        print(time_inference_generation)
        print(time_inference_generation/df_test.shape[0])
        wandb.log({"time_inference_generation": time_inference_generation})
        wandb.log({"per_q_time_inference_generation": time_inference_generation/df_test.shape[0]})
        
        return question_prediction_dict

    def calculate_metrics(self, df_test):
        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        # TODO: check if this way of calculating the time is correct
        if self.config.inference_device == 'cuda':
            torch.cuda.synchronize()
        question_pred_dict= self.inference(df_test)
        predicted_answers=[question_pred_dict[q_id][1] for q_id in df_test['question_id']]
        gold_answers=df_test['answer_text'].tolist()
        
        # TODO/DOUBT: should we add a classification filter first? 
        assert len(predicted_answers)==len(gold_answers)
        squad_f1_per_span = [compute_f1(predicted_answers[i],gold_answers[i])  for i in range(len(predicted_answers))]
        mean_squad_f1 = np.mean(squad_f1_per_span)

        classification_prediction = [1 if (len(predicted_answers[i]) != 0) else 0 for i in range(len(predicted_answers)) ] 
        classification_actual = df_test["answerable"].astype(int)
        classification_f1 = f1_score(classification_actual, classification_prediction)
        classification_accuracy = accuracy_score(classification_actual, classification_prediction)
        clf_report = classification_report(classification_actual, classification_prediction, output_dict=True)
        print(pd.DataFrame(clf_report).T)

        metrics = {
            "classification_accuracy": classification_accuracy,
            "clf_report": clf_report,
            "classification_f1": classification_f1,
            "mean_squad_f1": mean_squad_f1,
            # "mean_time_per_question (ms)": results["mean_time_per_question"]*1000,
        }

        wandb.log({"metrics": metrics})
        wandb.log({"predicted_answers": predicted_answers})
        wandb.log({"gold_answers": gold_answers})

        return metrics





class BiDAFTrainer():

    def __init__(self, config, model, optimizer, device, criterion, ques2idx, char2idx, word2idx, train_ds, retriever=None):
        self.config = config
        self.device = device

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.train_ds = train_ds

        wandb.watch(self.model)

        self.ques2idx = ques2idx
        self.char2idx = char2idx
        self.word2idx = word2idx

        self.retriever = retriever

    def _train_step(self, dataloader, epoch):
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}")

            out = self.model(batch)
            loss = (self.criterion(out[0], batch["start_positions"].long().to(self.device)) + self.criterion(out[1], batch["end_positions"].long().to(self.device))) / 2

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
                    
                out = self.model(batch)
                loss = (self.criterion(out[0], batch["start_positions"].long().to(self.device)) + self.criterion(out[1], batch["end_positions"].long().to(self.device))) / 2

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))
                wandb.log({"val_batch_loss": total_loss / (batch_idx + 1)})

        wandb.log({"val_epoch_loss": total_loss / (batch_idx + 1)})
        return (total_loss / (batch_idx + 1))


    def predict(self, batch):
        return self.model(batch)


    def inference(self, df_test):
        self.model.to(self.config.inference_device)
        self.device = self.config.inference_device
        self.model.device = self.config.inference_device

        title_id_list = df_test["title_id"].unique()
        gb_title = df_test.groupby("title_id")
        df_test_matched = pd.DataFrame()

        df_unique_con = df_test.drop_duplicates(subset=["context"])
        unmatched=0

        start_time = time.time()
        for title_id in title_id_list:
            df_temp = gb_title.get_group(title_id)
            
            for idx, row in df_temp.iterrows():
                question = row["question"]
                question_id = row["question_id"]
                context_id = row["context_id"]

                df_contexts=pd.DataFrame()
                if self.retriever is not None:
                    doc_idx_filtered, doc_text_filtered = self.retriever.retrieve_top_k(question, str(title_id), k=self.config.top_k)
                    df_contexts_og = df_unique_con.loc[df_unique_con["context_id"].isin([int(doc_idx) for doc_idx in doc_idx_filtered])].copy()
                    # TODO: we can endup sampling things in doc_idx_filtered again
                    df_contexts_random =  df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=max(0,self.config.top_k-len(doc_idx_filtered)),random_state=self.config.seed)
                    df_contexts = pd.concat([df_contexts_og, df_contexts_random], axis=0, ignore_index=True)
                else:
                    df_contexts = df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=self.config.top_k,random_state=self.config.seed)
                df_contexts.loc[:, "question"] = question
                df_contexts.loc[:, "question_id"] = question_id
                df_contexts.loc[:, "answerable"] = False
                df_contexts.loc[:, "answer_start"] = ""
                df_contexts.loc[:, "answer_text"] = ""
                
                original_context_idx = df_contexts.loc[df_contexts["context_id"] == context_id]
                if (len(original_context_idx) == 0):    
                    # print(f"original paragraph not in top k {unmatched}")
                    unmatched+=1
                    # don't uncomment this
                    # df_contexts = pd.concat([df_contexts, pd.DataFrame(row)], axis=0, ignore_index=True)
                else:

                    row_dict = row.to_dict()
                    del row_dict["context_ids"]
                    del row_dict["question_ids"]
                    
                    df_contexts.loc[df_contexts["context_id"] == context_id, row_dict.keys()] = row_dict.values()
                df_test_matched = pd.concat([df_test_matched, df_contexts], axis=0, ignore_index=True)

        # print(f"original paragraph not in top k {unmatched}")
        preprocess_df(df_test_matched)
        df_test_matched['context_ids'] = df_test_matched.context.parallel_apply(context_to_ids, word2idx=self.word2idx)
        df_test_matched['question_ids'] = df_test_matched.question.parallel_apply(question_to_ids, word2idx=self.word2idx)
        
        test_ds = BiDAF_Dataset(self.config, df_test_matched, self.char2idx, max_context_len = self.train_ds.max_context_len, max_word_ctx = self.train_ds.max_word_ctx, 
								max_question_len = self.train_ds.max_question_len, max_word_ques = self.train_ds.max_word_ques) #, hide_tqdm=True
        
        test_dataloader = DataLoader(test_ds, batch_size=self.config.data.val_batch_size, collate_fn=test_ds.collate_fn)
        time_test_dataloader_generation=1000*(time.time() - start_time)
        print(time_test_dataloader_generation)
        print(time_test_dataloader_generation/df_test.shape[0])
        wandb.log({"time_test_dataloader_generation": time_test_dataloader_generation})
        wandb.log({"per_q_time_test_dataloader_generation": time_test_dataloader_generation/df_test.shape[0]})
        
        print(f"{len(df_test_matched)=}")
        print(f"{len(df_temp)=}")
       
        start_time=time.time()
        question_prediction_dict={q_id:(0,"") for q_id in df_test_matched["question_id"].unique()}

        # TODO: without sequentional batch iteration
        for qp_batch_id, qp_batch in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
            
            # para, para_id, theme, theme_id, question, question_id
            pred = self.predict(qp_batch)
            if self.config.model.two_step_loss:
                confidence_scores=pred[1] # -> [32,1]
            else:
                start_probs=F.softmax(pred[0],dim=1)  # -> [32,512] 
                end_probs=F.softmax(pred[1],dim=1)    # -> [32,512] 

                max_start_probs=torch.max(start_probs, axis=1)  # -> [32,1] 
                max_end_probs=torch.max(end_probs,axis=1)       # -> [32,1]

                confidence_scores=max_end_probs.values*max_start_probs.values  # -> [32,1]                

            for batch_idx, q_id in enumerate(qp_batch["question_id"]):
                if (question_prediction_dict[q_id][0] < confidence_scores[batch_idx]):
                    # using the context in the qp_pair get extract the span using max_start_prob and max_end_prob                    
                    context = qp_batch["context"][batch_idx]
                    # offset_mapping = qp_batch["question_context_offset_mapping"][batch_idx]
                    decoded_answer = ""
                    
                    start_index = max_start_probs.indices[batch_idx].item()
                    end_index = max_end_probs.indices[batch_idx].item()

                    decoded_answer = context[start_index:end_index]

                    if len(decoded_answer) > 0:
                        question_prediction_dict[q_id] = (confidence_scores[batch_idx].item(), decoded_answer)

                    question_prediction_dict[q_id]=(confidence_scores[batch_idx].item(), decoded_answer)
    
        time_inference_generation=1000*(time.time()-start_time)
        print(time_inference_generation)
        print(time_inference_generation/df_test.shape[0])
        wandb.log({"time_inference_generation": time_inference_generation})
        wandb.log({"per_q_time_inference_generation": time_inference_generation/df_test.shape[0]})
        
        return question_prediction_dict

    def calculate_metrics(self, df_test):
        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        # TODO: check if this way of calculating the time is correct
        if self.config.inference_device == 'cuda':
            torch.cuda.synchronize()
        question_pred_dict= self.inference(df_test)
        predicted_answers=[question_pred_dict[q_id][1] for q_id in df_test['question_id']]
        gold_answers=df_test['answer_text'].tolist()
        
        # TODO/DOUBT: should we add a classification filter first? 
        assert len(predicted_answers)==len(gold_answers)
        squad_f1_per_span = [compute_f1(predicted_answers[i],gold_answers[i])  for i in range(len(predicted_answers))]
        mean_squad_f1 = np.mean(squad_f1_per_span)

        classification_prediction = [1 if (len(predicted_answers[i]) != 0) else 0 for i in range(len(predicted_answers)) ] 
        classification_actual = df_test["answerable"].astype(int)
        classification_f1 = f1_score(classification_actual, classification_prediction)
        classification_accuracy = accuracy_score(classification_actual, classification_prediction)
        clf_report = classification_report(classification_actual, classification_prediction, output_dict=True)
        print(pd.DataFrame(clf_report).T)

        metrics = {
            "classification_accuracy": classification_accuracy,
            "clf_report": clf_report,
            "classification_f1": classification_f1,
            "mean_squad_f1": mean_squad_f1,
            # "mean_time_per_question (ms)": results["mean_time_per_question"]*1000,
        }

        wandb.log({"metrics": metrics})
        wandb.log({"predicted_answers": predicted_answers})
        wandb.log({"gold_answers": gold_answers})

        return metrics
