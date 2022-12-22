import numpy as np 
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pytorch_lightning as pl
from . import Base_Model
from utils import compute_f1
from transformers import BartForConditionalGeneration
from tqdm import tqdm

class FewShotQA_Model(Base_Model):
    """
    DO NOT change the calculate_metrics function
    """

    def __init__(self, config, tokenizer):
        self.config = config

        self.model = BartForConditionalGeneration("facebook/bart-large")
        self.tokenizer = tokenizer

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.training.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs("fewshot_qa_outputs/", exist_ok=True)
        self.save_path = os.path.join("fewshot_qa_outputs/")
    
    def train(self, epoch, tokenizer, model, device, dataloader, optimizer):
        """
        Function to be called for training with the parameters passed from main function
        """
        model = model.to(device)
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

    def validate(self, epoch, tokenizer, model, device, val_dataloader, max_gen_length=32): # IMPORTANT TODO: Don't hardcode 32 
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

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    max_length=max_gen_length, 
                    num_beams=1, 
                    early_stopping=True,
                    #decoder_start_token_id=tokenizer.bos_token_id
                    )

                pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
                target_text = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in target]
                
                predictions.extend(pred_text)
                actuals.extend(target_text)
        
        return predictions, actuals

    def postprocess_preds(predictions):
        processed_preds = []
        for p in predictions:
            if 'answers' in p:
                processed_preds.append(p.split('answers')[1].split('context')[0].strip())
            else:
                processed_preds.append("")
        return processed_preds

    def postprocess_actuals(self, actuals):
        acts = []
        for ac in actuals:
            if 'answers' in ac:
                acts.append([a.split('answers')[1].split('context')[0].strip() for a in ac])
            else:
                acts.append("")
        return acts

    def __train__(self, train_dataloader):
        for epoch in range(self.config.training.epochs):
            self.train(epoch, self.tokenizer, self.model, self.device, train_dataloader, self.optimizer)

    def __inference__(self, test_dataloader):
        epoch=1
        predictions, actuals = self.validate(epoch, self.tokenizer, self.model, self.device, test_dataloader)
        processed_preds = self.postprocess_preds(predictions)
        processed_actuals = self.postprocess_actuals(actuals)
        
        generation_data = {"Generated_Text": predictions, "Actual_Text": actuals}
        generation_data["PP_Generated_Text"] = processed_preds
        generation_data["PP_Actual_Text"] = processed_actuals
        generation_df = pd.DataFrame(generation_data)

        generation_df.to_csv(os.path.join(self.save_path, "predictions.csv"), index=False)

        all_ground = []
        for batch_idx, batch in tqdm(enumerate(test_dataloader), position = 0, leave = True):
            all_ground.extend(batch["answerable"].detach().cpu().numpy())

        result = {"ground": all_ground,
                "predicted_spans": processed_preds,
                "gold_spans": processed_actuals}
            
        return result

    def few_shot_calculate_metrics(self, dataloader):

        """
            1. Run the inference script
            2. Calculate the time taken
            3. Calculate the F1 score
            4. Return all
        """

        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        results = self.__inference__(dataloader)
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time() * 1000)) - tsince
        # print ('test time elapsed {}ms'.format(ttime_elapsed))

        ttime_per_example = (ttime_elapsed * dataloader.batch_size)/len(results["ground"])

        f1_spans = []
        for i in range(len(results["predicted_spans"])):
            f1_spans.append(compute_f1(results["predicted_spans"][i], results["gold_spans"][i])) # For the text

        qa_f1 = np.mean(f1_spans)
        return qa_f1, ttime_per_example