
# dataloader - question, gold context id, gold answer
# retriever.process(question, theme_id, k ) -> k paragraphs sorted based on tf-idfs
# qa.process(question, k paragraphs) -> k (answers, confidence, paragraph)
# answer SquadF1, paragraph accuracy

from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


class CustomDataset(Dataset):

    def __init__(self, df):
      self.df = df

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      return {key: self.data[key][idx] for key in self.data.keys()}

    def collate_fn(self, items):
        keys = items[0].keys()
        batch = {key: torch.stack([x[key] for x in items], dim=0) for key in keys}
        return batch

def evaluate_endtoend(retriever, qa, dataset, tokenized_paragraphs, tokenizer, k=3):
    
    paragraph_ids_all = retriever.predict(dataset["question"], dataset["theme_id"], k=k)
    dataset["question_id"] = dataset.index
    # dataset["answers_predicted"] = []
    questions_tokenized = tokenizer(
            dataset["question"],
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True
        )
    dataset["question_input_ids"] = questions_tokenized.pop("input_ids")
    dataset["question_attention_mask"] = questions_tokenized.pop("attention_mask")

    new_dataset = pd.Dataframe(columns=["question_input_ids", "question_attention_mask", "context_input_ids", "context_attention_mask", "context_offset_mapping"])
    for question_idx in range(len(dataset)):
        paragraph_ids = paragraph_ids_all[question_idx]
        for i in paragraph_ids:
            paragraph = tokenized_paragraphs[i]
            new_row = {
                "question_input_ids": dataset["question_input_ids"][question_idx],
                "question_attention_mask": dataset["question_attention_mask"][question_idx],
                "context_input_ids": paragraph['input_ids'], 
                "context_attention_mask": paragraph['attention_mask'],
                "context_offset_mapping": paragraph['offset_mapping'],
                "question_id": question_idx,
                "paragraph_id": i
            }
            new_dataset.append(new_row)
      
    qp_dataset = CustomDataset(new_dataset)
    qp_dataloader = DataLoader(qp_dataset, batch_size=16)
    dataset["preds"] = [[] for _ in range(len(dataset))]
    for batch_idx, batch in enumerate(qp_dataloader):
        preds = qa.predict(batch)
        pred_start_index = torch.argmax(preds.start_logits, axis=1)
        pred_end_index = torch.argmax(preds.end_logits, axis=1)
        for idx in range(len(batch)):
            dataset["preds"][batch["question_id"][idx]].append(
                {
                    "pred_start_index": pred_start_index[idx],
                    "pred_end_index": pred_end_index[idx],
                    "pred_start_logit": preds.start_logits[idx][pred_start_index[idx]],
                    "pred_end_logit": preds.end_logits[idx][pred_end_index[idx]],
                    "paragraph_id": batch["paragraph_id"][idx]
                }
            )


    

        
    
    # create dataloader with question paragraph pairs and question id (q, p1) (q, p2), (q, p3) , (q2, p1) .... 
    # predict start token, end token of each question paragraph pair, extract string and store the string and confidence in a list for each question id
    # for each question id calculate metrics with gold answer


#     for batch_idx, batch in enumerate(dataloader):
#         theme_id = batch["theme_id"]
#         question = batch["question"]
#         answer = batch["answer"]

#         for paragraph_id in paragraph_ids:
#             # question = batch["question"][question_idx]
            
#             # # list of paragraph ids in the same theme as q
#             # q_para_ids = para_ids_batch[question_idx]

#             # # list of paragraphs for in the same theme as q
#             # paragraphs = list(set(dataset.df.iloc[q_para_ids]["Paragraph"]))

#             # # question id (primary key in df)
#             # q_id = batch["id"][question_idx]

#             # # create kb dataframe
#             # df_kb = self._create_inference_df(question, dataset, paragraphs, q_id)
#             # # print(q_para_ids)
#             # # print(len(df_kb))

#             # temp_ds = SQuAD_Dataset(dataset.config, df_kb, dataset.tokenizer)
#             # temp_dataloader = DataLoader(temp_ds, batch_size=dataset.config.data.val_batch_size, collate_fn=temp_ds.collate_fn)

#             total_time_per_question = 0
            
#             # loop for iterating over question para pairs to extract paras
#             for qp_batch_id, qp_batch in enumerate():
#                 start_time = time.time()
#                 paragraph_id_dict = tokenized_paragraphs[paragraph_id]
#                 pred = qa.predict(batch['question'], paragraph_id_dict["input_ids"])

#                 offset_mappings_list = paragraph_id_dict["offset_mapping"]
#                 # qp_batch["question_context_offset_mapping"]
#                 contexts_list = paragraph_id_dict["text"]
#                 # qp_batch["context"]

#                 gold_answers.extend([answer[0] if (len(answer) != 0) else "" for answer in qp_batch["answer"]])

#                 pred_start_index = torch.argmax(pred.start_logits, axis=1)
#                 pred_end_index = torch.argmax(pred.end_logits, axis=1)

#                 # iterate over each context
#                 for c_id, context in enumerate(contexts_list):
#                     # TODO: don't take only best pair (see HF tutorial)

#                     pred_answer = ""
#                     if (offset_mappings_list[c_id][pred_start_index[c_id]] is not None and offset_mappings_list[c_id][pred_end_index[c_id]]):
#                         try:
#                             pred_start_char = offset_mappings_list[c_id][pred_start_index[c_id]][0]
#                             pred_end_char = offset_mappings_list[c_id][pred_end_index[c_id]][1]
                        
#                         except:
#                             print(offset_mappings_list[c_id])
#                             raise ValueError

#                         pred_answer = context[pred_start_char:pred_end_char]

#                     predicted_answers.append(pred_answer)
                    
#                 # TODO: remove offset_mapping etc. lookup from inference time (current calculation is the absolute worst case time)
#                 total_time_per_question += (time.time() - start_time)
#                 total_time_per_question_list.append(total_time_per_question)



# def inference(self, dataset, dataloader):

#   # TODO: use only dataset (applying transforms as done in collate_fn here itself)
#   self.model.to(self.config.inference_device)
#   self.device = self.config.inference_device
#   self.model.device = self.config.inference_device

#   tepoch = tqdm(dataloader, unit="batch", position=0, leave=True)
#   tepoch.set_description("Inference Step")

#   total_time_per_question_list = []
#   predicted_answers = []
#   gold_answers = []

#   # TODO: is this time calculation correct?
#   for batch_idx, batch in enumerate(tepoch):
      
#       # list of titles in the batch 
#       title = batch["title"]  

#       # list of paragraph indices (in dataset.data) for each question in the batch
#       para_ids_batch = [dataset.theme_para_id_mapping[t] for t in title]    # List of para ids for each question in the batch

#       # iterate over questions in the batch
#       for question_idx in range(len(para_ids_batch)):
#           question = batch["question"][question_idx]
          
#           # list of paragraph ids in the same theme as q
#           q_para_ids = para_ids_batch[question_idx]

#           # list of paragraphs for in the same theme as q
#           paragraphs = list(set(dataset.df.iloc[q_para_ids]["Paragraph"]))

#           # question id (primary key in df)
#           q_id = batch["id"][question_idx]

#           # create kb dataframe
#           df_kb = self._create_inference_df(question, dataset, paragraphs, q_id)
#           # print(q_para_ids)
#           # print(len(df_kb))

#           temp_ds = SQuAD_Dataset(dataset.config, df_kb, dataset.tokenizer)
#           temp_dataloader = DataLoader(temp_ds, batch_size=dataset.config.data.val_batch_size, collate_fn=temp_ds.collate_fn)

#           total_time_per_question = 0
          
#           # loop for iterating over question para pairs to extract paras
#           for qp_batch_id, qp_batch in enumerate(temp_dataloader):
#               start_time = time.time()
#               pred = self.predict(qp_batch)

#               offset_mappings_list = qp_batch["question_context_offset_mapping"]
#               contexts_list = qp_batch["context"]

#               gold_answers.extend([answer[0] if (len(answer) != 0) else "" for answer in qp_batch["answer"]])

#               pred_start_index = torch.argmax(pred.start_logits, axis=1)
#               pred_end_index = torch.argmax(pred.end_logits, axis=1)

#               # iterate over each context
#               for c_id, context in enumerate(contexts_list):
#                   # TODO: don't take only best pair (see HF tutorial)

#                   pred_answer = ""
#                   if (offset_mappings_list[c_id][pred_start_index[c_id]] is not None and offset_mappings_list[c_id][pred_end_index[c_id]]):
#                       try:
#                           pred_start_char = offset_mappings_list[c_id][pred_start_index[c_id]][0]
#                           pred_end_char = offset_mappings_list[c_id][pred_end_index[c_id]][1]
                      
#                       except:
#                           print(offset_mappings_list[c_id])
#                           raise ValueError

#                       pred_answer = context[pred_start_char:pred_end_char]

#                   predicted_answers.append(pred_answer)
                  
#               # TODO: remove offset_mapping etc. lookup from inference time (current calculation is the absolute worst case time)
#               total_time_per_question += (time.time() - start_time)
#               total_time_per_question_list.append(total_time_per_question)

#   print(total_time_per_question_list)

#   results = {
#               "mean_time_per_question": np.mean(np.array(total_time_per_question_list)),
#               "predicted_answers": predicted_answers,
#               "gold_answers": gold_answers,
#           }     

#   return results
