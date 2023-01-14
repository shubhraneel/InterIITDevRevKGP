import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering

from pathlib import Path
from transformers.onnx import FeaturesManager
import transformers

class BaselineQA(nn.Module):
    def __init__(self, config, device):
        super(BaselineQA, self).__init__()

        self.config = config 
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model.model_path)
        if config.model.two_step_loss:
            self.score=nn.Linear(config.model.dim,1)
            self.loss_fct=nn.BCEWithLogitsLoss()
        if config.model.bio_tags:
            self.mlp_out = nn.Linear(config.model.dim, 3)

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
        if self.config.model.two_step_loss:
            cls_tokens=out.hidden_states[-1][:,0]
            scores=self.score(cls_tokens) # [32,1]
            out.loss+=self.loss_fct(scores,batch["answerable"])

            return (out,torch.nn.functional.softmax(scores))
        
        if self.config.model.bio_tags:
            ner_scores = self.mlp_out(out.hidden_states[-1])
            ner_scores_T = ner_scores.transpose(1, 2)
            loss = F.cross_entropy(batch['bio_tags'], ner_scores_T)
            out.loss = loss
            return (out,F.softmax(ner_scores, dim=2))

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
