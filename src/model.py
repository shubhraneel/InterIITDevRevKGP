import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering,AutoModelForSequenceClassification

from pathlib import Path
from transformers.onnx import FeaturesManager
import transformers

class BaselineQA(nn.Module):
    def __init__(self, config, device):
        super(BaselineQA, self).__init__()

        self.config = config 
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model.model_path)
        if config.model.clf_loss:
            self.score=nn.Linear(self.model.config.hidden_size,1)
            self.loss_fct=nn.BCEWithLogitsLoss()

        self.device = device

    def forward(self, batch):
        if not self.config.model.non_pooler:
            out = self.model(input_ids = batch["question_context_input_ids"].to(self.device), 
                            attention_mask = batch["question_context_attention_mask"].to(self.device),
                            token_type_ids = batch["question_context_token_type_ids"].to(self.device),
                            start_positions = batch["start_positions"].to(self.device),
                            end_positions = batch["end_positions"].to(self.device),
                            output_hidden_states=True)
        else:
            out = self.model(input_ids = batch["question_context_input_ids"].to(self.device), 
                            attention_mask = batch["question_context_attention_mask"].to(self.device),
                            start_positions = batch["start_positions"].to(self.device),
                            end_positions = batch["end_positions"].to(self.device),
                            output_hidden_states=True)
        # print("test",out.keys())
        if self.config.model.clf_loss:
            cls_tokens=out.hidden_states[-1][:,0]
            # start_probs=F.softmax(out.start_logits,dim=1)  # -> [32,512] 
            # end_probs=F.softmax(out.end_logits,dim=1)    # -> [32,512] 

            # max_start_probs=torch.max(start_probs, axis=1)  # -> [32,1] 
            # max_end_probs=torch.max(end_probs,axis=1)       # -> [32,1]

            # confidence_scores=end_probs.values*start_probs.values  # -> [32,1]
            # scores=confidence_scores.squeeze(1)
            scores=self.score(cls_tokens).squeeze(1) # [32]
            out.loss+=self.loss_fct(scores,batch["answerable"].to(self.device).float())

            return out

        return out  

    def export_to_onnx(self, tokenizer):
        # TODO Using torch.onnx.export
        # Will use transformers.onnx.export for transformer models

        # TODO Using transformers.onnx if this doesn't work
        feature = "question-answering"

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model, feature=feature)
        onnx_config = model_onnx_config(self.model.config)

        # export
        onnx_inputs, onnx_outputs = transformers.onnx.export(
                preprocessor=tokenizer,
                model=self.model,
                config=onnx_config,
                opset=13,
                output=Path("checkpoints/{}/model.onnx".format(self.config.load_path))
        )

        print(onnx_inputs, onnx_outputs)

class BaselineClf(nn.Module):
    def __init__(self, config, device):
        super(BaselineClf, self).__init__()

        self.config = config 
        self.model = AutoModelForSequenceClassification.from_pretrained(
              self.config.model.clf_model_path,
              num_labels=2,
            )

        self.device = device

    def forward(self, batch):
        if not self.config.model.non_pooler:
            out = self.model(input_ids = batch["question_context_input_ids"].to(self.device), 
                            attention_mask = batch["question_context_attention_mask"].to(self.device),
                            token_type_ids = batch["question_context_token_type_ids"].to(self.device),
                            output_hidden_states=True,
                            labels=batch["answerable"].to(self.device),
                          )
        else:
            out = self.model(input_ids = batch["question_context_input_ids"].to(self.device), 
                            attention_mask = batch["question_context_attention_mask"].to(self.device),
                            output_hidden_states=True,
                            labels=batch["answerable"].to(self.device),
                          )

        return out  

    ## Confirm that this works
    def export_to_onnx(self, tokenizer):
        # TODO Using torch.onnx.export
        # Will use transformers.onnx.export for transformer models

        # TODO Using transformers.onnx if this doesn't work
        feature = "question-answering"

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model, feature=feature)
        onnx_config = model_onnx_config(self.model.config)

        # export
        onnx_inputs, onnx_outputs = transformers.onnx.export(
                preprocessor=tokenizer,
                model=self.model,
                config=onnx_config,
                opset=13,
                output=Path("checkpoints/{}/model.onnx".format(self.config.load_path))
        )

        print(onnx_inputs, onnx_outputs)