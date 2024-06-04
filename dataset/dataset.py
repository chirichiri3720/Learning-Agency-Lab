import torch
from torch.utils.data import Dataset

# Dataset class
class EssayDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        if self.labels is not None:
            label = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

