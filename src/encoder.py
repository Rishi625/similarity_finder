
import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class Encoder:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_embeddings(self, texts,batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_encodings = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(**batch_encodings)

            sentence_embeddings = self._mean_pooling(batch_embeddings, batch_encodings['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            all_embeddings.extend(sentence_embeddings)
        return np.array([embedding.cpu().numpy() for embedding in all_embeddings])
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)