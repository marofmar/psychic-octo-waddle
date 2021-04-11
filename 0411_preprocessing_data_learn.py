# to automatically download the vocab used during pretraining,
# use the "from_pretrained()" method

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

# Base use
encoded_input = tokenizer("안녕, 난 사실 지금 배가 고파.")
print(encoded_input)

decoded_output = tokenizer.decode(encoded_input["input_ids"])
print(decoded_output)  # [CLS] 안녕, 난 사실 지금 배가 고파. [SEP]

encode_2 = tokenizer("아이야, 넌 이름이 뭐니?")
print(encode_2)

decode_2 = tokenizer.decode(encode_2["input_ids"])
print(decode_2)


tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")
encode_3 = tokenizer2("과연, 이게 될까?")
print(encode_3)
decode_3 = tokenizer.decode(encode_3["input_ids"])
print(decode_3)
