from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np

class BertExtractor:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
    

    def extract_features(self, texts, batch_size=32):
        features = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_encodings = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                batch_features = self.model(**batch_encodings).last_hidden_state[:, 0, :].cpu().numpy()
            features.append(batch_features)
        return np.concatenate(features)