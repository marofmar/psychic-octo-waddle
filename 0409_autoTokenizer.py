from transformers import AutoTokenizer, AutoModelForSequenceClassification
#model_name = "distilbert-base-uncased-finetuend-sst-2-english"
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# using the tokenizer
inputs = tokenizer("We are very happy to show you this amazing library.")

#print(inputs) # returns the ids of the tokens


# if I want to send a batch to my tokenizer,
# pt signifies 'pytorch'
pt_batch = tokenizer(
    ["We are happy to show you this amazing library.",
     "Have you seen the movie?"],
     padding=True,
     truncation=True,
     max_length=512,
     return_tensors="pt")


for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}")


pt_outputs = pt_model(**pt_batch)  # unpack the dictionary to feed the dictionary keys into the model

print(f"tensor output: {pt_outputs}")

import torch.nn.functional as F
pt_prediction = F.softmax(pt_outputs[0], dim = -1)  # Applying the SoftMax activation to get predictions

print(f"After SoftMax: {pt_prediction}")


# if I have labels, I can provide them to the model.
import torch
pt_outputs = pt_model(**pt_batch, labels=torch.tensor([1,0]))
print(f"torch output provided with label: {pt_outputs}")  # computed with LOSS


# Once my model is fine-tuned, I can save them in the following way. 
"""tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)"""

# I can ask the model to return all hidden states and all attention weights if I need them:
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = pt_outputs[-2:]



# Customizing the model: Config






