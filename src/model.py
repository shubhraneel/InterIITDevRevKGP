from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
# from allennlp.modules.span_extractors import EndpointSpanExtractor
from transformers import AutoModelForQuestionAnswering
from transformers.onnx import FeaturesManager


class BaselineQA(nn.Module):
    def __init__(self, config, device):
        super(BaselineQA, self).__init__()

        self.config = config
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.model.model_path
        )
        if config.model.two_step_loss:
            self.score = nn.Linear(config.model.dim, 1)
            self.loss_fct = nn.BCEWithLogitsLoss()
        elif config.model.span_level:
            self.span_extractor = EndpointSpanExtractor(
                input_dim=config.model.dim,
                num_width_embeddings=10,
                span_width_embedding_dim=config.model.dim,
            )
            seq_indices = list(range(self.config.data.answer_max_len))
            self.span_indices = [
                [x, y]
                for x in seq_indices
                for y in seq_indices
                if y - x >= 0 and y - x <= config.data.answer_max_len
            ]
            self.span_mlp = nn.Linear(config.model.dim * 3, 1)
            self.loss_fct = nn.BCEWithLogitsLoss()

        self.device = device

    def forward(self, batch):
        input = dict()
        input['input_ids'] = batch["question_context_input_ids"].to(self.device)
        input['attention_mask'] = batch["question_context_input_ids"].to(self.device)
        if not self.config.inference:
            input['start_positions'] = batch["start_positions"].to(self.device)
            input['end_positions'] = batch["end_positions"].to(self.device)
        if not self.config.model.non_pooler:
            input['token_type_ids'] = batch["question_context_token_type_ids"].to(self.device)
        print('input to the model')
        print(input)
        out = self.model(**input)
        if self.config.model.two_step_loss:
            cls_tokens = out.hidden_states[-1][:, 0]
            scores = self.score(cls_tokens)  # [32,1]
            out.loss += self.loss_fct(scores, batch["answerable"])

            return (out, torch.nn.functional.softmax(scores))

        elif self.config.model.span_level:
            token_embeddings = out.hidden_states[-1]
            span_indices = torch.tensor(
                [
                    self.span_indices
                    for _ in range(batch["question_context_input_ids"].shape[0])
                ]
            )
            span_indices = span_indices.to(self.device)
            span_embeddings = self.span_extractor(
                token_embeddings,
                span_indices,
                sequence_mask=batch["question_context_attention_mask"].to(self.device),
            )
            mlp_out = self.span_mlp(span_embeddings).squeeze(-1)
            # loss = F.cross_entropy(mlp_out, batch["span_indices"].to(self.device))
            binary_labels = F.one_hot(batch["span_indices"], num_classes=len(self.span_indices))
            loss = self.loss_fct(mlp_out, binary_labels.to(self.device))
            out.loss = loss
            return out, mlp_out

        return out

    def export_to_onnx(self, tokenizer, filename):
        # TODO Using torch.onnx.export
        # Will use transformers.onnx.export for transformer models

        # TODO Using transformers.onnx if this doesn't work
        feature = "question-answering"

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            self.model, feature=feature
        )
        onnx_config = model_onnx_config(self.model.config)

        # export
        onnx_inputs, onnx_outputs = transformers.onnx.export(
            preprocessor=tokenizer,
            model=self.model,
            config=onnx_config,
            opset=13,
            output=Path(filename),
        )

        print(onnx_inputs, onnx_outputs)
