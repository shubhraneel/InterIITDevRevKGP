import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import copy

from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification

from . import Base_Model
from tqdm import tqdm
import pandas as pd
from data import SQuAD_Dataset
from torch.utils.data import DataLoader
import time


class AutoModel_Classifier(pl.LightningModule):
    def __init__(self, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()

        self.config = config
        
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(config.model.model_path, num_labels=config.model.num_labels)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, batch):
        # if "answerable" in batch.keys():
            # out = self.classifier_model(input_ids=batch["question_paragraph_input_ids"], 
            #                         attention_mask=batch["question_paragraph_attention_mask"], 
            #                         token_type_ids=batch["question_paragraph_token_type_ids"],
            #                         labels=batch["answerable"],
            #                         )
        # else:
        # print(batch["question_context_input_ids"].size())
        if len(batch["question_context_input_ids"].size()) == 1:
            batch["question_context_input_ids"] = batch["question_context_input_ids"].unsqueeze(0)
            batch["question_context_attention_mask"] = batch["question_context_attention_mask"].unsqueeze(0)
            batch["question_context_token_type_ids"] = batch["question_context_token_type_ids"].unsqueeze(0)

        out = self.classifier_model(input_ids=batch["question_context_input_ids"], 
                                attention_mask=batch["question_context_attention_mask"], 
                                token_type_ids=batch["question_context_token_type_ids"],
                                )
    
        return out

    def training_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_context_input_ids"], 
                                    attention_mask=batch["question_context_attention_mask"], 
                                    token_type_ids=batch["question_context_token_type_ids"],
                                    labels=batch["answerable"],
                                    )
        self.log('train_loss_classifier', out.loss)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_context_input_ids"], 
                                    attention_mask=batch["question_context_attention_mask"], 
                                    token_type_ids=batch["question_context_token_type_ids"],
                                    labels=batch["answerable"],
                                    )
        self.log('val_loss_classifier', out.loss)
        return out.loss

    def test_step(self, batch, batch_idx):
        out = self.classifier_model(input_ids=batch["question_context_input_ids"], 
                                    attention_mask=batch["question_context_attention_mask"], 
                                    token_type_ids=batch["question_context_token_type_ids"],
                                    labels=batch["answerable"],
                                    )
        self.log('test_loss_classifier', out.loss)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer


class AutoModel_QA(pl.LightningModule):
    def __init__(self, config, train_dataloader = None, validation_dataloader = None, test_dataloader = None):
        super().__init__()

        self.config = config
        
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(config.model.model_path)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def forward(self, batch):

        # if "answer_encoded_start_idx" in batch.keys():
        #     out = self.qa_model(input_ids = batch["question_paragraph_input_ids"], 
        #                         attention_mask = batch["question_paragraph_attention_mask"],
        #                         token_type_ids = batch["question_paragraph_token_type_ids"],
        #                         start_positions = batch["answer_encoded_start_idx"],
        #                         end_positions = batch["answer_encoded_start_idx"],
        #                         )
        # else:
        if len(batch["question_context_input_ids"].size()) == 1:
            batch["question_context_input_ids"] = batch["question_context_input_ids"].unsqueeze(0)
            batch["question_context_attention_mask"] = batch["question_context_attention_mask"].unsqueeze(0)
            batch["question_context_token_type_ids"] = batch["question_context_token_type_ids"].unsqueeze(0)

        out = self.qa_model(input_ids = batch["question_context_input_ids"], 
                            attention_mask = batch["question_context_attention_mask"],
                            token_type_ids = batch["question_context_token_type_ids"],
                            )

        return out

    def training_step(self, batch, batch_idx):
        # TODO: pass answer start and end idx
        out = self.qa_model(input_ids = batch["question_context_input_ids"], 
                            attention_mask = batch["question_context_attention_mask"],
                            token_type_ids = batch["question_context_token_type_ids"],
                            start_positions = batch["start_positions"],
                            end_positions = batch["end_positions"],
                            )
        self.log('train_loss_qa', out.loss)
        
        # TODO: ANSWERS CONVERGING TO 0, 0
        # print("Actual spans")
        # print(batch["start_positions"])
        # print(batch["end_positions"])

        # print("Predicted spans")
        # print(torch.argmax(out.start_logits, dim=1))
        # print(torch.argmax(out.end_logits, dim=1))

        return out.loss
    
    def validation_step(self, batch, batch_idx):
        out = self.qa_model(input_ids = batch["question_context_input_ids"], 
                            attention_mask = batch["question_context_attention_mask"],
                            token_type_ids = batch["question_context_token_type_ids"],
                            )
        self.log('val_loss_qa', out.loss)
        return out.loss

    def test_step(self, batch, batch_idx):
        out = self.qa_model(input_ids = batch["question_context_input_ids"], 
                            attention_mask = batch["question_context_attention_mask"],
                            token_type_ids = batch["question_context_token_type_ids"],
                            )
        self.log('test_loss_qa', out.loss)
        return out.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.lr)
        return optimizer


class AutoModel_Classifier_QA(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """
    def __init__(self, config, tokenizer = None, logger=None):
        self.config = config
        self.logger = logger

        self.classifier_trainer = pl.Trainer(max_epochs = self.config.training.epochs, accelerator = "gpu", devices = 1, logger=logger)
        self.classifier_model = AutoModel_Classifier(self.config)

        self.qa_model_trainer = pl.Trainer(max_epochs = self.config.training.epochs, accelerator = "gpu", devices = 1, logger=logger)
        self.qa_model = AutoModel_QA(self.config)

        self.tokenizer = tokenizer

        self.config = config
        
    def __train__(self, dataloader):
        print("Starting training")

        self.classifier_trainer.fit(model = self.classifier_model, train_dataloaders = dataloader)
        self.qa_model_trainer.fit(model = self.qa_model, train_dataloaders = dataloader)

    def _create_inference_df(self, question, dataset, paragraphs, id):
        
        # TODO: Optimize by replacing loc with query function 
        df_kb = pd.DataFrame()
        for p in paragraphs:
            # if (q, p) is already in the df, just return the answer
            df_in_data = dataset.df.loc[(dataset.df["Question"] == question) & (dataset.df["Paragraph"] == p)]
            if (len(df_in_data) != 0):
                assert len(df_in_data) == 1
                df_kb = pd.concat([df_kb, df_in_data], axis = 0).reset_index(drop = True)
            else:
                # Unnamed: 0	Theme	Paragraph	Question	Answer_possible	Answer_text	Answer
                # kb_dict = {"Unnamed: 0": id, "Question": question, ""}
                kb_row = dataset.df.loc[dataset.df["Unnamed: 0"] == id].to_dict(orient = "records")[0]
                kb_row["Paragraph"] = p
                kb_row["Answer_possible"] = False
                kb_row["Answer_start"] = []
                kb_row["Answer_text"] = []

                # df_in_data = pd.DataFrame(kb_row, index = False)

                df_kb = df_kb.append(kb_row, ignore_index = True)

                # df_kb = pd.concat([df_kb, df_in_data], axis = 0).reset_index(drop = True)
                # kb_row["Question"] = question
                # return unanswerable (Answerable_possible = False, Answer_text = [], Answer_start = [])

        return df_kb
    
    def __inference__(self, dataset, dataloader, logger):

        all_preds = []
        all_ground = []

        all_start_preds = []
        all_end_preds = []
        all_start_ground = []
        all_end_ground = []
        all_input_words = []

        predicted_spans = []
        gold_spans = []

        time_list = []

        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):

            title = batch["title"]
            para_ids = [dataset.theme_para_id_mapping[t] for t in title]    # List of para ids for each question in the batch

            # TODO: Optimize this 
            # Loop for iterating over each question
            time_per_question = []

            for i in range(len(para_ids)):
                para_id = para_ids[i]
                
                question = batch["question"][i]
                paragraphs = list(set(dataset.df.iloc[para_id]["Paragraph"]))
                id = batch["id"][i]
                df_kb = self._create_inference_df(question, dataset, paragraphs, id)
                
                # one question =====
                temp_squad = SQuAD_Dataset(dataset.config, df_kb, dataset.tokenizer)
                temp_dataloader = DataLoader(temp_squad, batch_size = dataset.config.data.val_batch_size, collate_fn = temp_squad.collate_fn)

                t = 0
                st_time = time.time()
                # Loop for iterating over question para pairs to extract paras
                q_all_preds = []
                q_all_ground = []
                
                extracted_paragraphs = []
                for qp_batch_id, qp_batch in tqdm(enumerate(temp_dataloader), position = 0, leave = True):
                
                    pred = self.classifier_model.predict_step(qp_batch, qp_batch_id)
                    t += time.time() - st_time

                    all_preds.extend(torch.argmax(pred.logits, axis = 1).tolist())
                    all_ground.extend(qp_batch["answerable"].detach().cpu().numpy())
                    
                    # Too much load in metric calculation
                    # Discuss how to calculate metric

                    # Extract the paragraphs for which answerable is true, also store the logits for confidence
                    #  
                    st_time = time.time()

                    # ERROR in the line below -- TypeError: 'int' object is not iterable            fix in the morning :)))) 
                    x = (torch.argmax(pred.logits, axis = 1) == 1).nonzero().squeeze().tolist()
                    if not isinstance(x, list):
                        x = [x]

                    extracted_paragraphs.extend([qp_batch["context"][i] for i in x])
                    t += time.time() - st_time

                    # q_all_preds.extend(pred.logits.tolist())
                    # q_all_ground.extend(qp_batch["answerable"].detach().cpu().numpy())

                df_extracted = self._create_inference_df(question, dataset, extracted_paragraphs, id)
                
                # DOUBT: Should we pick just one paragraph, or all the paragraphs for which model predicted True? 

                extracted_squad = SQuAD_Dataset(dataset.config, df_extracted, dataset.tokenizer)
                extracted_dataloader = DataLoader(extracted_squad, batch_size = dataset.config.data.val_batch_size, collate_fn = extracted_squad.collate_fn)
                
                q_all_start_preds = []
                q_all_end_preds = []
                q_all_start_ground = []
                q_all_end_ground = []
                q_all_input_words = []

                for ex_batch_idx, ex_batch in tqdm(enumerate(extracted_dataloader), position = 0, leave = True):
                    st_time = time.time()
                    pred = self.qa_model.predict_step(ex_batch, ex_batch_idx)
                    t += time.time() - st_time

                    q_all_start_preds.extend(torch.argmax(pred.start_logits, axis = 1).tolist())
                    q_all_end_preds.extend(torch.argmax(pred.end_logits, axis = 1).tolist())
                    q_all_start_ground.extend(batch["start_positions"].detach().cpu().numpy())
                    q_all_end_ground.extend(batch["end_positions"].detach().cpu().numpy())
                    
                    # NOTE: Have to write a postprocessing function to join two words in the list, such as ['digest', '##ive'] should be one word
                    q_all_input_words.extend(self.tokenizer.batch_decode(sequences = ex_batch["context_input_ids"]))

                q_predicted_spans = []
                q_gold_spans = []

                print("\n")
                print(len(q_all_start_preds))
                print(len(q_all_end_preds))
                print(len(q_all_input_words))

                if not isinstance(q_all_input_words[0], list):
                    q_all_input_words = [q_all_input_words]
                # print(q_all_input_words)

                for idx, sentence in enumerate(q_all_input_words):
                    sentence = sentence.split(" ")
                    print(sentence)
                    print(len(sentence))
                    
                    st_time = time.time()
                    predicted_span = " ".join(sentence[q_all_start_preds[idx]: q_all_end_preds[idx]])
                    t += time.time() - st_time
                    
                    gold_span = " ".join(sentence[q_all_start_ground[idx]: q_all_end_ground[idx]])

                    q_predicted_spans.append(predicted_span)
                    q_gold_spans.append(gold_span)


                time_list.append(t)
                all_start_preds.extend(q_all_start_preds)
                all_end_preds.extend(q_all_end_preds)
                all_start_ground.extend(q_all_start_ground)
                all_end_ground.extend(q_all_end_ground)

                predicted_spans.extend(q_predicted_spans)
                gold_spans.extend(q_gold_spans)


        #     pred = self.classifier_model.predict_step(batch, batch_idx)
        #     all_preds.extend(torch.argmax(pred.logits, axis = 1).tolist())
        #     all_ground.extend(batch["answerable"].detach().cpu().numpy())

        # all_start_preds = []
        # all_end_preds = []
        # all_start_ground = []
        # all_end_ground = []
        # all_input_words = []

        # for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
        #     pred = self.qa_model.predict_step(batch, batch_idx)
        #     all_start_preds.extend(torch.argmax(pred.start_logits, axis = 1).tolist())
        #     all_end_preds.extend(torch.argmax(pred.end_logits, axis = 1).tolist())
        #     all_start_ground.extend(batch["start_positions"].detach().cpu().numpy())
        #     all_end_ground.extend(batch["end_positions"].detach().cpu().numpy())
            
        #     all_input_words.extend(self.tokenizer.batch_decode(sequences = batch["context_input_ids"]))

        # predicted_spans = []
        # gold_spans = []

        # for idx, sentence in enumerate(all_input_words):
        #     sentence = sentence.split(" ")
        #     predicted_span = " ".join(sentence[all_start_preds[idx]: all_end_preds[idx]])
        #     gold_span = " ".join(sentence[all_start_ground[idx]: all_end_ground[idx]])

        #     predicted_spans.append(predicted_span)
        #     gold_spans.append(gold_span)

        result = {"preds": all_preds,
                "ground": all_ground,
                "all_start_preds": all_start_preds,
                "all_end_preds": all_end_preds,
                "all_start_ground": all_start_ground,
                "all_end_ground": all_end_ground,
                "all_input_words": all_input_words,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans,
                "inference_time": np.mean(time_list)}
            
        return result

    def __evaluate__(self, dataloader):
        # TODO
        pass