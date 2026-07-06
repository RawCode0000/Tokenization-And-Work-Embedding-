from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "Artificial Intelligence is changing the world."

tokens = tokenizer.tokenize(sentence)

print("Original Sentence:")
print(sentence)

print("\nTokens:")
print(tokens)