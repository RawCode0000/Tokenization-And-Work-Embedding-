from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

sentence = "Hello नमस्ते Bonjour こんにちは"

tokens = tokenizer.tokenize(sentence)

print(sentence)

print(tokens)