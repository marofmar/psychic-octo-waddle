from transformers import AutoTokenizer, AutoModel, AdamW

## ---- train mode
model = AutoModel.from_pretrained("beomi/KcELECTRA-base")
model.train()

#optimizer = AdamW(model.parameters(), lr=1e-5)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr = 1e-5



## ---- training batch
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
text_batch = ["나는 픽사, 디즈니 좋아한다!", "나는 픽사, 디즈니 몰라."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

print(encoding)

