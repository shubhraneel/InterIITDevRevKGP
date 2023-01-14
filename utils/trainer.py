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
from data import SQuAD_Dataset

import onnxruntime
from onnxruntime.quantization import quantize_dynamic

from transformers.modeling_outputs import QuestionAnsweringModelOutput

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class Trainer():
    def __init__(self, config, model, optimizer, device, 
        tokenizer, ques2idx, df_val=None,val_retriever=None):
        self.config = config
        self.device = device

        self.tokenizer = tokenizer

        self.optimizer = optimizer
        self.model = model


        wandb.watch(self.model)

        self.ques2idx = ques2idx

        self.df_val = df_val
        self.val_retriever = val_retriever

        self.prepared_test_loader=None
        self.prepared_test_df_matched=None
        # setup onnx runtime if config.onnx is true
        self.onnx_runtime_session = None
        if (self.config.ONNX):
            self.model.export_to_onnx(tokenizer)
            # TODO Handle this case when using quantization without ONNX using torch.quantization
        
            if (self.config.quantize):
                quantize_dynamic("checkpoints/{}/model.onnx".format(self.config.load_path), "checkpoints/{}/model_quantized.onnx".format(self.config.load_path))
            
            sess_options = onnxruntime.SessionOptions()
            # TODO Find if this line helps
            # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            model_path = "checkpoints/{}/model.onnx".format(self.config.load_path) if not self.config.quantize else "checkpoints/{}/model_quantized.onnx".format(self.config.load_path)
            self.onnx_runtime_session = onnxruntime.InferenceSession(model_path, sess_options)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0-total1)**2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss

    def contrastive_adaptive_loss(self, outputs, batch):
        input_ids = batch["question_context_input_ids"].to(self.device)
        attention_mask = batch["question_context_attention_mask"].to(self.device)
        token_type_ids = batch["question_context_token_type_ids"].to(self.device)
        start_positions = batch["start_positions"].to(self.device)
        end_positions = batch["end_positions"].to(self.device)

        input_type = None
        if "input_type" in batch.keys() and batch["input_type"] != None:
            input_type = batch['input_type'].to(self.device)
        else:
            input_type = torch.randint(0, 2, (input_ids.shape[0],), dtype=torch.long).to(self.device)
        
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]

        sequence_output = outputs['hidden_states'][-1]

        a_mask_1 = torch.zeros(token_type_ids.shape[0], token_type_ids.shape[1]+1).to(token_type_ids.device)
        a_mask_1[torch.arange(a_mask_1.shape[0]), start_positions] = 1
        a_mask_1 = a_mask_1.cumsum(dim=1)[:, :-1]
        a_mask_2 = torch.zeros(token_type_ids.shape[0], token_type_ids.shape[1]+1).to(token_type_ids.device)
        a_mask_2[torch.arange(a_mask_2.shape[0]), end_positions+1] = 1
        a_mask_2 = a_mask_2.cumsum(dim=1)[:, :-1]
        a_mask = a_mask_1 * (1 - a_mask_2)

        splits = (input_ids == 102) * torch.arange(input_ids.shape[1], 0, -1).to(input_ids.device)
        _, splits = torch.sort(splits, -1, descending=True)
        splits = splits[:, :2]
        # splits = (input_ids == 102).nonzero()[:, 1].reshape(input_ids.size(0),-1)
        c_mask = (token_type_ids == 1) * attention_mask
        c_mask[torch.arange(c_mask.size(0)), splits[:, 0]] = 0
        c_mask[torch.arange(c_mask.size(0)), splits[:, 1]] = 0
        c_mask = c_mask * (1 - a_mask)

        q_mask = (token_type_ids == 0) * attention_mask
        q_mask[torch.arange(q_mask.size(0)), splits[:, 0]] = 0
        q_mask[:, 0] = 0

        a_rep = (sequence_output * a_mask.unsqueeze(-1)).sum(1) / a_mask.sum(-1).unsqueeze(-1)
        cq_mask = ((c_mask + q_mask) > 0) * 1.0
        cq_rep = (sequence_output * cq_mask.unsqueeze(-1)).sum(1) / cq_mask.sum(-1).unsqueeze(-1)

        can_loss = -1*self.mmd(cq_rep, a_rep)

        if len((input_type==0).nonzero()[:, 0]) != 0 and len((input_type==1).nonzero()[:, 0]) != 0:
            a_rep_source = a_rep[(input_type==0).nonzero()[:, 0]].view(-1, a_rep.size(1))
            a_rep_target = a_rep[(input_type==1).nonzero()[:, 0]].view(-1, a_rep.size(1))
            cq_rep_source = cq_rep[(input_type==0).nonzero()[:, 0]].view(-1, cq_rep.size(1))
            cq_rep_target = cq_rep[(input_type==1).nonzero()[:, 0]].view(-1, cq_rep.size(1))

            can_loss += self.mmd(a_rep_source, a_rep_target) + self.mmd(cq_rep_source, cq_rep_target)
        
        return can_loss

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
            if batch_idx%300==0:
              self.log_ipop_batch(batch,out,batch_idx)

            loss = out.loss
            if self.config.training.can_loss and not self.config.model.non_pooler:
                loss += self.contrastive_adaptive_loss(out, batch) * self.config.training.can_loss_beta

            if loss.isnan():
              print("NanSense", batch_idx)
              self.optimizer.zero_grad()
              batch_1 = {}
              for i in range(28):
                for key in batch.keys():
                  batch_1[key] = batch[key][i:i+4]
                with torch.no_grad():
                  out = self.model(batch_1)
                if self.config.training.can_loss and not self.config.model.non_pooler:
                  out.loss += self.contrastive_adaptive_loss(out, batch_1) * self.config.training.can_loss_beta
                if out.loss.isnan():
                  print(batch_1)
                else:
                  print(out.loss)
              self.optimizer.zero_grad()
              continue
            
            loss.backward()

            total_loss += loss.item()
            tepoch.set_postfix(loss = total_loss / (batch_idx + 1))
            wandb.log({"train_batch_loss": total_loss / (batch_idx + 1)})

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        wandb.log({"train_epoch_loss": total_loss / (batch_idx + 1)})

        return (total_loss / (batch_idx + 1))

    def log_ipop_batch(self,batch,out,batch_idx):
      rows=[]
      for i in range(len(batch["context"])):
        context = batch["context"][i]
        start_probs=F.softmax(out.start_logits,dim=1)
        end_probs=F.softmax(out.end_logits,dim=1)

        max_start_probs=torch.max(start_probs, axis=1)
        max_end_probs=torch.max(end_probs, axis=1)

        offset_mapping = batch["question_context_offset_mapping"][i]
        start_index = max_start_probs.indices[i].item()
        end_index = max_end_probs.indices[i].item()
        decoded_answer=""
        if (offset_mapping[start_index] is not None and offset_mapping[end_index] is not None): 
          start_char = offset_mapping[start_index][0]
          end_char = offset_mapping[end_index][1]

          decoded_answer = context[start_char:end_char]
        
        tgt_answer=""
        if (offset_mapping[batch["start_positions"][i]] is not None and offset_mapping[batch["end_positions"][i]] is not None):
          tgt_start=offset_mapping[batch["start_positions"][i]][0]
          tgt_end=offset_mapping[batch["end_positions"][i]][1]

          tgt_answer= context[tgt_start:tgt_end]
        
        answer=batch["answer"][i]
        question=batch["question"][i]

        rows.append([question,answer,decoded_answer,tgt_answer])
      
      my_table = wandb.Table(columns=["question","dataset answer","predicted answer", "train target"], data=rows)
      wandb.log({"Batch "+str(batch_idx)+ " IP/OP": my_table})

    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        for epoch in range(self.config.training.epochs):
            self._train_step(train_dataloader, epoch)
            
            if ((val_dataloader is not None) and (((epoch + 1) % self.config.training.evaluate_every)) == 0):
                # self.evaluate(val_dataloader)
                if epoch==0:
                  self.calculate_metrics(self.df_val,self.val_retriever,'val',self.device,do_prepare=True)
                else:
                  self.calculate_metrics(self.df_val,self.val_retriever,'val',self.device,do_prepare=False)
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

                if self.config.training.can_loss and self.config.model.non_pooler:
                    loss += self.contrastive_adaptive_loss(out, batch) * self.config.training.can_loss_beta

                loss = out.loss

                total_loss += loss.item()
                tepoch.set_postfix(loss = total_loss / (batch_idx+1))
                wandb.log({"val_batch_loss": total_loss / (batch_idx + 1)})

        wandb.log({"val_epoch_loss": total_loss / (batch_idx + 1)})
        return (total_loss / (batch_idx + 1))


    def predict(self, batch):

        if (self.config.ONNX):
            # Set up inputs for the ONNX Runtime Invocation
            ort_inputs = None
            if not self.config.model.non_pooler:
                ort_inputs = {
                    self.onnx_runtime_session.get_inputs()[0].name: to_numpy(batch['question_context_input_ids']), 
                    self.onnx_runtime_session.get_inputs()[1].name: to_numpy(batch['question_context_attention_mask']),
                    self.onnx_runtime_session.get_inputs()[2].name: to_numpy(batch['question_context_token_type_ids']),
                    # self.onnx_runtime_session.get_inputs()[3].name: to_numpy(batch['start_positions'].unsqueeze(dim=1)),
                    # self.onnx_runtime_session.get_inputs()[4].name: to_numpy(batch['end_positions'].unsqueeze(dim=1))
                }

            else:
                ort_inputs = {
                    self.onnx_runtime_session.get_inputs()[0].name: to_numpy(batch['question_context_input_ids']), 
                    self.onnx_runtime_session.get_inputs()[1].name: to_numpy(batch['question_context_attention_mask']),
                    # self.onnx_runtime_session.get_inputs()[2].name: to_numpy(batch['start_positions'].unsqueeze(dim=1)),
                    # self.onnx_runtime_session.get_inputs()[3].name: to_numpy(batch['end_positions'].unsqueeze(dim=1))
                }

            # print(ort_inputs)

            ort_outputs = self.onnx_runtime_session.run(None, ort_inputs)

            # print(ort_outputs)
            out = QuestionAnsweringModelOutput(
                # loss = torch.tensor(ort_outputs[0]),
                start_logits = torch.tensor(ort_outputs[0]),
                end_logits = torch.tensor(ort_outputs[1])
            )

            return out
        
        return self.model(batch)

    def prepare_df_before_inference(self, df_test,retriever,prefix,device):
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
                if retriever is not None:
                    doc_idx_filtered, doc_text_filtered = retriever.retrieve_top_k(question, str(title_id), k=self.config.top_k)
                    df_contexts_og = df_unique_con.loc[df_unique_con["context_id"].isin([int(doc_idx) for doc_idx in doc_idx_filtered])].copy()
                    # TODO: we can endup sampling things in doc_idx_filtered again
                    df_contexts_random =  df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=max(0,self.config.top_k-len(doc_idx_filtered)),random_state=self.config.seed)
                    df_contexts = pd.concat([df_contexts_og, df_contexts_random], axis=0, ignore_index=True)
                else:
                    df_contexts =  df_unique_con.loc[df_unique_con['title_id']==title_id].sample(n=self.config.top_k,random_state=self.config.seed)
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
        wandb.log({"time_"+str(prefix)+"_dataloader_generation": time_test_dataloader_generation})
        wandb.log({"per_q_"+str(prefix)+"_time_"+str(prefix)+"_dataloader_generation": time_test_dataloader_generation/df_test.shape[0]})
        
        print(f"{len(df_test_matched)=}")
        print(f"{len(df_temp)=}")

        self.prepared_test_loader=test_dataloader
        self.prepared_test_df_matched=df_test_matched


    def inference(self, df_test,retriever,prefix,device,do_prepare):
        self.model.to(device)
        self.device = device
        self.model.device = device

        if do_prepare:
          self.prepare_df_before_inference(df_test,retriever,prefix,device)
       
        start_time=time.time()
        question_prediction_dict={q_id:(0,"") for q_id in self.prepared_test_df_matched["question_id"].unique()}

        # TODO: without sequentional batch iteration
        for qp_batch_id, qp_batch in tqdm(enumerate(self.prepared_test_loader),total=len(self.prepared_test_loader)):
            
            if (len(qp_batch['question_context_input_ids'].shape) == 1):
                qp_batch['question_context_input_ids'] = qp_batch['question_context_input_ids'].unsqueeze(dim=0)
                qp_batch['question_context_attention_mask'] = qp_batch['question_context_attention_mask'].unsqueeze(dim=0)
                if not self.config.model.non_pooler:
                    qp_batch['question_context_token_type_ids'] = qp_batch['question_context_token_type_ids'].unsqueeze(dim=0)
            
            # para, para_id, theme, theme_id, question, question_id
            pred = self.predict(qp_batch)

            # print(pred.start_logits.shape) -> [32,512] 
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
                    if(len(decoded_answer)>0):
                        question_prediction_dict[q_id]=(confidence_scores[batch_idx].item(),decoded_answer)
    
        time_inference_generation=1000*(time.time()-start_time)
        print(time_inference_generation)
        print(time_inference_generation/df_test.shape[0])
        wandb.log({prefix+"_time_inference_generation": time_inference_generation})
        wandb.log({prefix+"_per_q_time_inference_generation": time_inference_generation/df_test.shape[0]})
        
        return question_prediction_dict

    def calculate_metrics(self, df_test,retriever,prefix,device,do_prepare):
        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        # TODO: check if this way of calculating the time is correct
        if device == 'cuda':
            torch.cuda.synchronize()
        question_pred_dict= self.inference(df_test,retriever,prefix,device,do_prepare)
        predicted_answers=[question_pred_dict[q_id][1] for q_id in df_test['question_id']]
        gold_answers=df_test['answer_text'].tolist()
        
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