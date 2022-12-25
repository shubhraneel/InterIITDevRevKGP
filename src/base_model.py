import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_f1
import time
import sklearn
from tqdm import tqdm
import torch
import os
import torch.onnx
from argparse import Namespace
import yaml
from yaml.loader import SafeLoader
from onnxruntime.transformers import optimizer as onnx_optimizer
import time
import psutil
import onnxruntime
import numpy



class Base_Model():
    def __init__(self):
        pass

    def __train__(self):
        raise NotImplementedError("No training method implemented")

    def __evaluate__(self):
        raise NotImplementedError("No evaluation method implemented")

    def __inference__(self):
        raise NotImplementedError("No inference method implemented")

    def calculate_metrics(self, dataloader):

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
        classification_f1 = sklearn.metrics.f1_score(results["preds"], results["ground"]) # For paragraph search task

        f1_spans = []
        for i in range(len(results["predicted_spans"])):
            f1_spans.append(compute_f1(results["predicted_spans"][i], results["gold_spans"][i])) # For the text

        qa_f1 = np.mean(f1_spans)
        return classification_f1, qa_f1, ttime_per_example

    def quantize(self, dataloader, model_type ,path_to_onnx_model_class,path_to_onnx_model_qa, attention_heads, hidden_states):
        
        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            break
      
        def export_onnx_model(inp, model, onnx_model_path,o_names):
            with torch.no_grad():
                inputs = {'input_ids' : inp["question_context_input_ids"], 
                            'attention_mask' : inp["question_context_attention_mask"],
                            'token_type_ids' : inp["question_context_token_type_ids"]}
                outputs = model(**inputs)

                symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
                torch.onnx.export(model,                                            # model being run
                            (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                            inputs['attention_mask'], 
                            inputs['token_type_ids']),                                         # model input (or a tuple for multiple inputs)
                            onnx_model_path,                                # where to save the model (can be a file or file-like object)
                            opset_version=11,                                 # the ONNX version to export the model to
                            do_constant_folding=True,                         # whether to execute constant folding for optimization
                            input_names=['input_ids',                         # the model's input names
                                        'attention_mask', 
                                        'token_type_ids'],
                            output_names=o_names,                    # the model's output names
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                        'attention_mask' : symbolic_names,
                                        'token_type_ids' : symbolic_names})

        model_class = self.classifier_model.classifier_model
        o_names_class = ['preds']
        export_onnx_model(batch, model_class,"model_classifier.onnx",o_names_class)
        optimized_model = onnx_optimizer.optimize_model("model_classifier.onnx", model_type=model_type, num_heads=attention_heads, hidden_size=hidden_states)
        optimized_model.convert_float_to_float16()
        optimized_fp16_model_path_class = "model_classifier_fp16.onnx"
        optimized_model.save_model_to_file(optimized_fp16_model_path_class)

        model_qa = self.qa_model.qa_model
        o_names_qa = ['start_preds','end_preds']
        export_onnx_model(batch, model_qa,"model_qa.onnx",o_names_qa)
        optimized_model = onnx_optimizer.optimize_model("model_qa.onnx", model_type=model_type, num_heads=attention_heads, hidden_size=hidden_states)
        optimized_model.convert_float_to_float16()
        optimized_fp16_model_path_qa = "model_qa_fp16.onnx"
        optimized_model.save_model_to_file(optimized_fp16_model_path_qa)
     
        optimized_fp16_model_path_qa = "model_qa_fp16.onnx"
        optimized_fp16_model_path_class = "model_classifier_fp16.onnx"
        
        sess_options_class = onnxruntime.SessionOptions()
        export_model_path = optimized_fp16_model_path_class
        sess_options_class.optimized_model_filepath = path_to_onnx_model_class
        sess_options_class.intra_op_num_threads=psutil.cpu_count(logical=True)
        EP_list = ['CPUExecutionProvider']
        session_class = onnxruntime.InferenceSession(export_model_path, sess_options_class, providers=EP_list)
        
        sess_options_qa = onnxruntime.SessionOptions()
        export_model_path = optimized_fp16_model_path_qa
        sess_options_qa.optimized_model_filepath = path_to_onnx_model_qa
        sess_options_qa.intra_op_num_threads=psutil.cpu_count(logical=True)
        EP_list = ['CPUExecutionProvider']
        session_qa = onnxruntime.InferenceSession(export_model_path, sess_options_qa, providers=EP_list)

     
        torch.cuda.synchronize()
        tsince = int(round(time.time() * 1000))
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
                device = "cpu"
                
                
        all_preds = []
        all_ground = []
        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            data = [batch["question_context_input_ids"],batch["question_context_attention_mask"],  batch["question_context_token_type_ids"],
                        batch['answerable'],batch['start_positions'],batch['end_positions'],batch['context_input_ids']]
                
            ort_inputs = {
                    'input_ids':      data[0].cpu().numpy(),
                    'attention_mask': data[1].cpu().numpy(),
                    'token_type_ids': data[2].cpu().numpy()
                }

            pred = np.array(session_class.run(None, ort_inputs))
            pred = np.argmax(pred, axis = 1).tolist()
            all_preds.extend(pred)
            all_ground.extend(batch["answerable"].detach().cpu().numpy())
        all_preds = np.array(all_preds)
        all_preds = all_preds.reshape(1,all_preds.size)
        all_preds = all_preds.tolist()
        all_preds = all_preds[0]
        
       
        all_start_preds = []
        all_end_preds = []
        all_start_ground = []
        all_end_ground = []
        all_input_words = []

        for batch_idx, batch in tqdm(enumerate(dataloader), position = 0, leave = True):
            data = [batch["question_context_input_ids"],batch["question_context_attention_mask"],  batch["question_context_token_type_ids"],
                        batch['answerable'],batch['start_positions'],batch['end_positions'],batch['context_input_ids']]
                
            ort_inputs = {
                    'input_ids':      data[0].cpu().numpy(),
                    'attention_mask': data[1].cpu().numpy(),
                    'token_type_ids': data[2].cpu().numpy()
                }
            pred = np.array(session_qa.run(None, ort_inputs))
            pred_0 = np.argmax(pred[0], axis = 1).tolist()
            pred_1 = np.argmax(pred[1], axis = 1).tolist()
            all_start_preds.extend(pred_0)
            all_end_preds.extend(pred_1)
            all_start_ground.extend(batch["start_positions"].detach().cpu().numpy())
            all_end_ground.extend(batch["end_positions"].detach().cpu().numpy())
            
            all_input_words.extend(self.tokenizer.batch_decode(sequences = batch["context_input_ids"]))
        all_start_preds = np.array(all_start_preds)
        all_start_preds = all_start_preds.reshape(1,all_start_preds.size)
        all_start_preds = all_start_preds.tolist()
        all_start_preds = all_start_preds[0]

        all_end_preds = np.array(all_end_preds)
        all_end_preds = all_end_preds.reshape(1,all_end_preds.size)
        all_end_preds = all_end_preds.tolist()
        all_end_preds = all_end_preds[0]
        #print(all_start_preds,all_start_ground)
        #print(all_end_preds, all_end_ground)
        predicted_spans = []
        gold_spans = []

        for idx, sentence in enumerate(all_input_words):
            sentence = sentence.split(" ")
            predicted_span = " ".join(sentence[all_start_preds[idx]: all_end_preds[idx]])
            gold_span = " ".join(sentence[all_start_ground[idx]: all_end_ground[idx]])

            predicted_spans.append(predicted_span)
            gold_spans.append(gold_span)

        results = {"preds": all_preds,
                "ground": all_ground,
                "all_start_preds": all_start_preds,
                "all_end_preds": all_end_preds,
                "all_start_ground": all_start_ground,
                "all_end_ground": all_end_ground,
                "all_input_words": all_input_words,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans}
                
        torch.cuda.synchronize()
        ttime_elapsed = int(round(time.time() * 1000)) - tsince
        onnx_ttime = (ttime_elapsed * dataloader.batch_size)/len(results["ground"])  
        #print(results['preds'],results['ground'])    
        onnx_class_f1 = sklearn.metrics.f1_score(results["preds"], results["ground"]) # For paragraph search task

        f1_spans = []
        for i in range(len(results["predicted_spans"])):
            f1_spans.append(compute_f1(results["predicted_spans"][i], results["gold_spans"][i])) # For the text

        onnx_qa_f1 = np.mean(f1_spans)
        
        
        os.remove("model_classifier.onnx")
        os.remove("model_classifier_fp16.onnx")
        os.remove("model_qa.onnx")
        os.remove("model_qa_fp16.onnx")
        return onnx_class_f1,onnx_qa_f1,onnx_ttime