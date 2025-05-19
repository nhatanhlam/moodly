import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
import pandas as pd

# 1. Load labeled_data.csv
df = pd.read_csv("labeled_data.csv")
texts = df["text"].tolist()
labels = df["label"].map({"tệ":0,"cũng ổn":1,"bình thường":2,"tốt":3}).tolist()

# 2. Tokenizer + Dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
class MoodDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k:torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = MoodDataset(texts, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 3. Model + optimizer + loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 4. Training loop
model.train()
for epoch in range(15):
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

# 5. Save
model.save_pretrained("moodly_bert_pt")
tokenizer.save_pretrained("moodly_bert_pt")
