## Model inputs

# input IDs

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sequence = "A Titan RTX has 24GB of VRAM"

tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)

inputs = tokenizer(sequence) # converted into IDs for models' understanding

# the token indices are under the key "input_ids"
encoded_sequence = inputs["input_ids"]
print(encoded_sequence)

# decode the previous sequence of ids
decoded_sequence = tokenizer.decode(encoded_sequence)
print(decoded_sequence)




## Attention mask
# this optional arg indicates which tokens should be attended to or not.

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."


encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]

print(len(encoded_sequence_a), len(encoded_sequence_b))  # 8 and 19
# the encoded versions have different length


# op1. pad up the first sentence a
# op2. trancate down the second one to the length of the first one

# op1.
padded_sequence = tokenizer([sequence_a, sequence_b], padding=True)
print(padded_sequence["input_ids"])


# attention mask is a binary tensor indicating the position of the padded indices
print(padded_sequence["attention_mask"])




## Token Type IDs
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"

encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
print(decoded)
print(encoded_dict['token_type_ids'])













## Position IDs




