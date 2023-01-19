import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering

from pathlib import Path
from transformers.onnx import FeaturesManager
import transformers
from collections import OrderedDict

# Hardcoding the Feature Object to output hidden_states as well
transformers.onnx.config.OnnxConfig._tasks_to_common_outputs['question-answering'] = OrderedDict(
    {
        "hidden_states": {0: "seq_len", 1: "batch", 2: "sequence", 3: "hidden_size"},
        "start_logits": {0: "batch", 1: "sequence"},
        "end_logits": {0: "batch", 1: "sequence"},
    }
)

class BaselineQA(nn.Module):
    def __init__(self, config, device):
        super(BaselineQA, self).__init__()

        self.config = config 
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model.model_path, output_hidden_states=True)
        if config.model.two_step_loss:
            self.score=nn.Linear(config.model.dim,1)
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
        if self.config.model.two_step_loss:
            cls_tokens=out.hidden_states[-1][:,0]
            scores=self.score(cls_tokens) # [32,1]
            out.loss+=self.loss_fct(scores,batch["answerable"])

            return (out,torch.nn.functional.softmax(scores))

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
