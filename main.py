"""Проверка окружения: версия torch, доступные GPU и поддерживаемая точность."""

import torch

print("Torch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("bf16 поддерживается:", torch.cuda.is_bf16_supported())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1024 ** 3:.1f} ГБ")
else:
    print("Обучение пойдёт на CPU (очень медленно) — на Kaggle включи GPU-ускоритель.")
