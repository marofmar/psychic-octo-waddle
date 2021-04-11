from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",        # outside of a named entity
    "B-MISC",   # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",   # Miscellaneous entity
    "B-PER",    # Beginning of a person's name right after another person's name
    "I-PER",    # Person's name
    "B-ORG",    # Beginning of an organization right after another organization
    "I-ORG",    # Organization
    "B-LOC",    # Beginning of a location right after another location
    "I-LOC",    # Location
]
sequence = "The portion of the lesion at the gray matter aspect is no enhancing,"\
"whereas the white matter aspect has an enhancing rim on the diffusion sequence,"\
"there was no diffusion restriction."

tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)

print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])
