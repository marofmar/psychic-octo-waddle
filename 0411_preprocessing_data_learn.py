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
print(decode_3)  # 안 됨 !


batch_sentences = ["옛날 옛날에 귀여운 고양이 한 마리가 살았어요.",
                   "그 고양이는 세상에서 제일 귀엽고 영특한 고양이.",
                   "바로 김후추."]
from pprint import pprint
encoded_inputs = tokenizer(batch_sentences)
pprint(encoded_inputs)

batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
pprint(batch)


for i in batch["input_ids"]:
    print(tokenizer.decode(i))
    # 캬캬 아주 잘 된다 !



## Processing pairs of sentences

pair_input = tokenizer("고양이는 몇 살 인가요?", "후추는 2살 입니다.")
print(pair_input)

print(tokenizer.decode(pair_input["input_ids"]))


batch_sentences = ["안녕, 나는 고양이야.",
                   "그리고, 나는 또 다른 고양이",
                   "마지막으로 나는 인간."]

batch_of_second_sentences = ["나는 첫 번째 고양이 후추.",
                             "그리고 여기는 두 번째 고양이 콜라.",
                             "마지막으로 인간 한 명은 이름 안 알려줄거다."]

encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
#pprint(encoded_inputs)

for ids in encoded_input["input_ids"]:
    print(tokenizer.decode(ids))

batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="pt")
for ids in batch["input_ids"]:
    print(tokenizer.decode(ids))

