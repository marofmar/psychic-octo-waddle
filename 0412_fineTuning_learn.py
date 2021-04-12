from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.train()  # train mode (default: eval)

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_batch=["I like Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(loss)
loss.backward()
optimizer.step()
scheduler.step()
