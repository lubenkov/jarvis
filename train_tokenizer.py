from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[UNK]", "<|endoftext|>"],
    show_progress=True,
)


print("Обучаю токенизатор...")
tokenizer.train(files=["data/raw/dialogues.txt"], trainer=trainer)

tokenizer.save("tokenizer.json")
print("Токенизатор сохранён в tokenizer.json")


test_text = "Привет, как дела?"
encoded = tokenizer.encode(test_text)
print("Исходный текст:", test_text)
print("Токены (кусочки текста):", encoded.tokens)
print("ID токенов (числа, которые увидит модель):", encoded.ids)
print("Раскодировано обратно:", tokenizer.decode(encoded.ids))