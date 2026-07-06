from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

print("Loading WikiText dataset...")

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_text = dataset["train"]["text"]

with open("wiki.txt", "w", encoding="utf-8") as f:
    for line in train_text:
        f.write(line + "\n")

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=5000)

tokenizer.train(["wiki.txt"], trainer)

sentence = "Artificial Intelligence is changing the world."

output = tokenizer.encode(sentence)

print("\nOriginal Sentence:")
print(sentence)

print("\nBPE Tokens:")
print(output.tokens)