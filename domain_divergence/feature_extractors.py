from transformers import AutoTokenizer, AutoModel
import torch

class transformer_tokenizer():
    def __init__(self, tokenizer_path, device):

        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def extract(self, text):

        return self.tokenizer(text, max_length=512, truncation="longest_first", return_offsets_mapping=True, padding='max_length', return_tensors="pt")["input_ids"]

    def create_feature_tensor(self, texts):
        
        all_features = []

        for text in texts:
            all_features.append(self.extract(text))

        all_features = torch.cat(all_features, dim=0)
        all_features = all_features.to(self.device)

        return all_features


class transformer_hidden_states():
    def __init__(self, model_path, device):

        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)

    def extract(self, text):

        return self.tokenizer(text, max_length=512, truncation="longest_first", return_offsets_mapping=True, padding='max_length', return_tensors="pt")["input_ids"]

    def create_feature_tensor(self, texts):
        
        all_features = []

        for text in texts:
            all_features.append(self.extract(text))

        all_features = torch.cat(all_features, dim=0).to(self.device)

        with torch.no_grad():
            all_features = self.model(all_features)[0]

        return all_features