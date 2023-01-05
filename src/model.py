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

        # with torch.no_grad():
        #     inputs = None
        #     input_names = None
        #     dynamic_axes_dict = None
        #     if not self.config.model.non_pooler:
        #         input_names = ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']
        #         input_ids = torch.ones(self.config.data.val_batch_size, self.config.data.max_length, dtype = torch.int64).to(self.config.inference_device)
        #         attention_mask = torch.ones(self.config.data.val_batch_size, self.config.data.max_length, dtype = torch.int64).to(self.config.inference_device)
        #         token_type_ids = torch.ones(self.config.data.val_batch_size, self.config.data.max_length, dtype = torch.int64).to(self.config.inference_device)
        #         start_positions = torch.ones(self.config.data.val_batch_size, 1,dtype = torch.int64).to(self.config.inference_device)
        #         end_positions = torch.ones(self.config.data.val_batch_size,1,dtype = torch.int64).to(self.config.inference_device)
        #         inputs = (input_ids, attention_mask, token_type_ids, start_positions, end_positions)
        #         # symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        #         # symbolic_name_positions = {0: 'batch_size'}
        #         # dynamic_axes_dict = {
        #         #     'input_ids': symbolic_names,        # variable length axes
        #         #     'attention_mask' : symbolic_names,
        #         #     'token_type_ids' : symbolic_names,
        #         #     'start_positions': symbolic_name_positions,
        #         #     'end_positions': symbolic_name_positions
        #         # }
        #         inputs_dict = {
        #             'input_ids': input_ids,
        #             'attention_mask': attention_mask,
        #             'token_type_ids': token_type_ids,
        #             'start_positions': start_positions,
        #             'end_positions': end_positions
        #         }

        #     else:
        #         input_names = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
        #         input_ids = torch.ones(self.config.data.val_batch_size, self.config.data.max_length, dtype = torch.int64).to(self.config.inference_device)
        #         attention_mask = torch.ones(self.config.data.val_batch_size, self.config.data.max_length, dtype = torch.int64).to(self.config.inference_device)
        #         start_positions = torch.ones(self.config.data.val_batch_size,1, dtype = torch.int64).to(self.config.inference_device)
        #         end_positions = torch.ones(self.config.data.val_batch_size, 1,dtype = torch.int64).to(self.config.inference_device)
        #         inputs = (input_ids, attention_mask, start_positions, end_positions)
        #         # symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        #         # symbolic_name_positions = {0: 'batch_size'}
        #         # dynamic_axes_dict = {
        #         #     'input_ids': symbolic_names,        # variable length axes
        #         #     'attention_mask' : symbolic_names,
        #         #     'start_positions': symbolic_name_positions,
        #         #     'end_positions': symbolic_name_positions
        #         # }
        #         inputs_dict = {
        #             'input_ids': input_ids,
        #             'attention_mask': attention_mask,
        #             'start_positions': start_positions,
        #             'end_positions': end_positions
        #         }

        #     outputs = self.model(**inputs_dict)

        #     print(outputs)

        #     torch.onnx.export(
        #         self.model,                                               # model being run
        #         inputs,                                                         # model input (or a tuple for multiple inputs)
        #         self.config.path_to_onnx_model,                                 # where to save the model (can be a file or file-like object)                                 # the ONNX version to export the model to
        #         do_constant_folding=True,                                       # whether to execute constant folding for optimization
        #         input_names=input_names,
        #         output_names=['start_logits', 'end_logits'],            # the model's output names
        #         # dynamic_axes=dynamic_axes_dict
        #         operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        #     )

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
                output=Path(self.config.path_to_onnx_model)
        )

        print(onnx_inputs, onnx_outputs)
