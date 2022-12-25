import torch
import torch.nn as nn


class BERTTokensModel(torch.nn.Module):
    def __init__(self, hidden_size, max_seq_len):
        super(BERTTokensModel, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.lin = nn.Linear(self.max_seq_len, self.hidden_size)

        self.activation = nn.Tanh()

        self.dense_intent = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout_intent = torch.nn.Dropout(0.2)
        self.classifier_intent = nn.Linear(self.hidden_size, 2)

    def forward(self, ids = None, mask = None, token_type_ids = None):


        output_1 = self.lin(ids)

        output_2 = self.dense_intent(output_1)
        output_intent = self.activation(output_2)
        output_intent = self.dropout_intent(output_intent)
        output_intent = self.classifier_intent(output_intent)

        return output_intent