from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')

set_seed(123)
a = generator("On my way home, I met a pig. After that, ", max_length = 20, num_return_sequences=10)

print(a)

