from transformers.models.bert.modeling_bert import BertPreTrainedModel,BertModel
from transformers import AutoModel,AutoConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
from typing import List, Optional, Tuple, Union

class BoostedBertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        bert_config=AutoConfig.from_pretrained(config.model.model_path)
        super().__init__(bert_config)

        self.bert = AutoModel.from_pretrained(config.model.model_path)
        self.num_labels = self.bert.config.num_labels
        
        
        self.qa_outputs = nn.ModuleList()
        for i in range(config.num_learners):
            self.qa_outputs.append(nn.Linear(bert_config.hidden_size, bert_config.num_labels))

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        batch,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        learner_id: Optional[int]=0,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = batch["question_context_input_ids"].to(self.bert.device)
        attention_mask = batch["question_context_attention_mask"].to(self.bert.device)
        token_type_ids = batch["question_context_token_type_ids"].to(self.bert.device)
        start_positions = batch["start_positions"].to(self.bert.device)
        end_positions = batch["end_positions"].to(self.bert.device)
        output_hidden_states=True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs[learner_id](sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )