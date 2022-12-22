import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from . import Base_Model
from fewshot_qa_utils import *

from transformers import BartForConditionalGeneration

class FewShotQA_Model(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """

    def __init__(self, config, tokenizer):
        self.config = config

        self.model = BartForConditionalGeneration("facebook/bart-large")
        self.tokenizer = tokenizer

        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.config.training.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("fewshot_qa_outputs/", exist_ok=True)
        self.save_path = os.path.join("fewshot_qa_outputs/")
    
    def train(epoch, tokenizer, model, device, dataloader, optimizer):
        """
        Function to be called for training with the parameters passed from main function
        """

        model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader, 0):
            target = batch["fewshot_qa_answer_input_ids"].to(device, dtype=torch.long)
            target[target == tokenizer.pad_token_id] = -100
            
            input_ids = batch["fewshot_qa_prompt_input_ids"].to(device, dtype=torch.long)
            attention_mask = batch["fewshot_qa_prompt_attention_mask"].to(device, dtype=torch.long)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target,
            )
            
            loss = outputs[0]
            total_loss += loss.item()            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(str(epoch), str(step), str(total_loss / (step + 1)))

    def validate(epoch, tokenizer, model, device, val_dataloader):
        """
        Function to evaluate model for predictions

        """
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, batch in enumerate(val_dataloader, 0):
                target = batch["fewshot_qa_answer_input_ids"].to(device, dtype=torch.long)
            
                input_ids = batch["fewshot_qa_prompt_input_ids"].to(device, dtype=torch.long)
                attention_mask = batch["fewshot_qa_prompt_attention_mask"].to(device, dtype=torch.long)

                # IMPORTANT TODO: Don't hardcode 32 
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    max_length=32, 
                    num_beams=1, 
                    early_stopping=True,
                    #decoder_start_token_id=tokenizer.bos_token_id
                    )

                pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target_text = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in target]
                
                predictions.extend(pred_text)
                actuals.extend(target_text)
        
        return predictions, actuals

    def __train__(self, train_dataloader, val_dataloader):
        best_f1 = -1
        for epoch in range(config.training.epochs):
            train(epoch, self.tokenizer, self.model, self.device, train_dataloader, self.optimizer)
            
            predictions, actuals = validate(epoch, self.tokenizer, self.model, self.device, val_dataloader)
            
            processed_preds = postprocess_preds(predictions)
            processed_actuals = postprocess_actuals(actuals)

            # IMPORTANT TODO: change this to our calculate_metrics function and delete the utils file (also in main)
            cur_metrics = get_metrics(processed_preds, processed_actuals)

            if cur_metrics[0] > best_f1:
                print(f"New best: {cur_metrics}")
                print(f"[Saving Model]...\n")
                
                # Saving the model after training

                model.save_pretrained(self.save_path)
                tokenizer.save_pretrained(self.save_path)
                best_f1 = cur_metrics[0]

    def __evaluate__(self, test_dataloader):
        predictions, actuals = validate(epoch, self.tokenizer, self.model, self.device, val_dataloader)
        processed_preds = postprocess_preds(predictions)
        processed_actuals = postprocess_actuals(actuals)
        
        generation_data = {"Generated_Text": predictions, "Actual_Text": actuals}
        generation_data["PP_Generated_Text"] = processed_preds
        generation_data["PP_Actual_Text"] = processed_actuals
        generation_df = pd.DataFrame(generation_data)

        generation_df.to_csv(os.path.join(self.save_path, "predictions.csv"), index=False)

        test_metrics = get_metrics(processed_preds, processed_actuals)
        print(f"Test metrics: {test_metrics}")