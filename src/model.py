import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
import numpy as np

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


class CharacterEmbeddingLayer(nn.Module):
    
    def __init__(self, char_vocab_dim, char_emb_dim, num_output_channels, kernel_size):
        
        super().__init__()
        
        self.char_emb_dim = char_emb_dim
        self.char_embedding = nn.Embedding(char_vocab_dim, char_emb_dim, padding_idx=1)
        self.char_convolution = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x = [bs, seq_len, word_len]
        # returns : [batch_size, seq_len, num_output_channels]
        # the output can be thought of as another feature embedding of dim 100.
        
        batch_size = x.shape[0]
        
        x = self.dropout(self.char_embedding(x))
        # x = [bs, seq_len, word_len, char_emb_dim]
        
        # following three operations manipulate x in such a way that
        # it closely resembles an image. this format is important before 
        # we perform convolution on the character embeddings.
        
        x = x.permute(0,1,3,2)
        # x = [bs, seq_len, char_emb_dim, word_len]
        
        x = x.view(-1, self.char_emb_dim, x.shape[3])
        # x = [bs*seq_len, char_emb_dim, word_len]
        
        x = x.unsqueeze(1)
        # x = [bs*seq_len, 1, char_emb_dim, word_len]
        
        # x is now in a format that can be accepted by a conv layer. 
        # think of the tensor above in terms of an image of dimension
        # (N, C_in, H_in, W_in).
        
        x = self.relu(self.char_convolution(x))
        # x = [bs*seq_len, out_channels, H_out, W_out]
        
        x = x.squeeze()
        # x = [bs*seq_len, out_channels, W_out]
                
        x = F.max_pool1d(x, x.shape[2]).squeeze()
        # x = [bs*seq_len, out_channels, 1] => [bs*seq_len, out_channels]
        
        x = x.view(batch_size, -1, x.shape[-1])
        # x = [bs, seq_len, out_channels]
        # x = [bs, seq_len, features] = [bs, seq_len, 100]
        
        
        return x

class HighwayNetwork(nn.Module):
    
    def __init__(self, input_dim, num_layers=2):
        
        super().__init__()
        
        self.num_layers = num_layers
        
        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        
        for i in range(self.num_layers):
            
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))
            
            x = gate_value * flow_value + (1-gate_value) * x
        
        return x

class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.highway_net = HighwayNetwork(input_dim)
        
    def forward(self, x):
        # x = [bs, seq_len, input_dim] = [bs, seq_len, emb_dim*2]
        # the input is the concatenation of word and characeter embeddings
        # for the sequence.
        
        highway_out = self.highway_net(x)
        # highway_out = [bs, seq_len, input_dim]
        
        outputs, _ = self.lstm(highway_out)
        # outputs = [bs, seq_len, emb_dim*2]
        
        return outputs

class BiDAF(nn.Module):
    
    def __init__(self, char_vocab_dim, emb_dim, char_emb_dim, num_output_channels, 
                 kernel_size, ctx_hidden_dim, device):
        '''
        char_vocab_dim = len(char2idx)
        emb_dim = 100
        char_emb_dim = 8
        num_output_chanels = 100
        kernel_size = (8,5)
        ctx_hidden_dim = 100
        '''
        super().__init__()
        
        self.device = device
        self.word_embedding = self.get_glove_embedding()
        self.character_embedding = CharacterEmbeddingLayer(char_vocab_dim, char_emb_dim, 
                                                      num_output_channels, kernel_size)
        self.contextual_embedding = ContextualEmbeddingLayer(emb_dim*2, ctx_hidden_dim)
        self.dropout = nn.Dropout()
        self.similarity_weight = nn.Linear(emb_dim*6, 1, bias=False)
        self.modeling_lstm = nn.LSTM(emb_dim*8, emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2)
        self.output_start = nn.Linear(emb_dim*10, 1, bias=False)
        self.output_end = nn.Linear(emb_dim*10, 1, bias=False)
        self.end_lstm = nn.LSTM(emb_dim*2, emb_dim, bidirectional=True, batch_first=True)

    def get_glove_embedding(self):
        
        weights_matrix = np.load('bidafglove.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device),freeze=True)

        return embedding

    def forward(self, batch):
        # ctx = [bs, ctx_len]
        # ques = [bs, ques_len]
        # char_ctx = [bs, ctx_len, ctx_word_len]
        # char_ques = [bs, ques_len, ques_word_len]
        
        ctx = batch["padded_context"]
        ques = batch["padded_question"]
        char_ctx = batch["char_context"]
        char_ques = batch["char_question"]

        ctx_len = ctx.shape[1]
        
        ques_len = ques.shape[1]
        
        ## GET WORD AND CHARACTER EMBEDDINGS
        
        ctx_word_embed = self.word_embedding(ctx)
        # ctx_word_embed = [bs, ctx_len, emb_dim]
        
        ques_word_embed = self.word_embedding(ques)
        # ques_word_embed = [bs, ques_len, emb_dim]
        
        ctx_char_embed = self.character_embedding(char_ctx)
        # ctx_char_embed =  [bs, ctx_len, emb_dim]
        
        ques_char_embed = self.character_embedding(char_ques)
        # ques_char_embed = [bs, ques_len, emb_dim]
        
        ## CREATE CONTEXTUAL EMBEDDING
        
        ctx_contextual_inp = torch.cat([ctx_word_embed, ctx_char_embed],dim=2)
        # [bs, ctx_len, emb_dim*2]
        
        ques_contextual_inp = torch.cat([ques_word_embed, ques_char_embed],dim=2)
        # [bs, ques_len, emb_dim*2]
        
        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)
        # [bs, ctx_len, emb_dim*2]
        
        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)
        # [bs, ques_len, emb_dim*2]
        
        
        ## CREATE SIMILARITY MATRIX
        
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1,1,ques_len,1)
        # [bs, ctx_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1,ctx_len,1,1)
        # [bs, 1, ques_len, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        
        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, emb_dim*2]
        
        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, emb_dim*6]
        
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]
        
        
        ## CALCULATE CONTEXT2QUERY ATTENTION
        
        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]
        
        c2q = torch.bmm(a, ques_contextual_emb)
        # [bs] ([ctx_len, ques_len] X [ques_len, emb_dim*2]) => [bs, ctx_len, emb_dim*2]
        
        
        ## CALCULATE QUERY2CONTEXT ATTENTION
        
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        # [bs, ctx_len]
        
        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]
        
        q2c = torch.bmm(b, ctx_contextual_emb)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, emb_dim*2]) => [bs, 1, emb_dim*2]
        
        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, emb_dim*2]
        
        ## QUERY AWARE REPRESENTATION
        
        G = torch.cat([ctx_contextual_emb, c2q, 
                       torch.mul(ctx_contextual_emb,c2q), 
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)
        
        # [bs, ctx_len, emb_dim*8]
        
        
        ## MODELING LAYER
        
        M, _ = self.modeling_lstm(G)
        # [bs, ctx_len, emb_dim*2]
        
        ## OUTPUT LAYER
        
        M2, _ = self.end_lstm(M)
        
        # START PREDICTION
        
        p1 = self.output_start(torch.cat([G,M], dim=2))
        # [bs, ctx_len, 1]
        
        p1 = p1.squeeze()
        # [bs, ctx_len]
        
        #p1 = F.softmax(p1, dim=-1)

        # END PREDICTION
        
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        # [bs, ctx_len, 1] => [bs, ctx_len]
        
        #p2 = F.softmax(p2, dim=-1)
        
        return p1, p2