import torch  # библиотека, с которой будем работать почти всё время

# torch.cuda.is_available() - проверяет, видит ли PyTorch вообще видеокарту через CUDA
print("CUDA доступна:", torch.cuda.is_available())

# Если видеокарта найдена - выведем её имя, чтобы убедиться что это именно MX450
if torch.cuda.is_available():
    print("Название GPU:", torch.cuda.get_device_name(0))
    # Сколько видеопамяти PyTorch считает доступной (в гигабайтах)
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("Видеопамять:", round(total_memory_gb, 2), "ГБ")