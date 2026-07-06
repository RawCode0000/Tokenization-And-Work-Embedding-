from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

words = [
    "playing",
    "unbelievable",
    "tokenization",
    "friendliness"
]

for word in words:
    print(word, "->", tokenizer.tokenize(word))