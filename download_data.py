
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

MAX_DOCUMENTS = 100000

with open("input.txt", "w", encoding="utf-8") as f:

    for i, item in enumerate(tqdm(dataset)):
        text = item["text"]

        if text and text.strip():
            f.write(text)
            f.write("\n\n")

        if i + 1 >= MAX_DOCUMENTS:
            break

print("Dataset extraction completed!")
