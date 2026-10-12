from datasets import load_dataset  # функция для скачивания датасетов с HuggingFace
import os
os.makedirs("data/raw", exist_ok=True)

print("Скачиваю датасет...")
dataset = load_dataset("inkoziev/jokes_dialogues", split="train")


print("Пример записи:", dataset[0])
print("Всего записей:", len(dataset))


lines = []
for example in dataset:

    context = example["context"]
    utterance = example["utterance"]
    text = context + "\n" + utterance
    lines.append(text)

print("Собрано текстовых блоков:", len(lines))


with open("data/raw/dialogues.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n<|endoftext|>\n")

print("Готово! Файл сохранён в data/raw/dialogues.txt")