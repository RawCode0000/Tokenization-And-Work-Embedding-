from datasets import load_dataset
from transformers import BertTokenizer

print("Loading IMDB Dataset...")

dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

sentence = dataset["train"][0]["text"]

tokens = tokenizer.tokenize(sentence)

print("\nOriginal Sentence:\n")
print(sentence[:300])

print("\nWordPiece Tokens:\n")
print(tokens[:100])