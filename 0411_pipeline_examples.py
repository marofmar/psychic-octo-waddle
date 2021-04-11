## Pipelines, Direct model use
# all tasks presented here leverage pre-trained checkpoints


## Sequence Classification
# - clssifying sequence according to a given number of classes
# - positive, negative

from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


## Sequence classification: Pharaphrases of each other?
# 1. instantiate a tokenizer and a model from the checkpoint name
# 2. build a sequence from the two sentences
# 3. pass this sequence through the model 0: not paraphrase, 1: is a paraphrase
# 4. compute the softmax and get probabilities over the classes
# 5. print the result

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace headquarters are situated in Manhattan"  # "TypeError: Can't convert this to PyBool..... 

paraphrase = tokenizer(sequence_0, sequence_1, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, sequence_2, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i]*100))}%")

for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i]*100))}%")

