import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, {props.total_memory / 1024 ** 3:.1f} GB")
else:
    print("Training will run on CPU (very slow) — enable a GPU accelerator on Kaggle.")
