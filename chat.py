import torch
from tokenizers import Tokenizer

from generate import load_model, generate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    model = load_model(device)

    print("Мини-Джарвис готов к общению. Для выхода напиши exit.")

    while True:
        user_input = input("Ты: ")
        if user_input.strip().lower() == "exit":
            break

        prompt = user_input.strip() + "\n"
        response = generate(model, tokenizer, prompt, device)

        reply = response[len(prompt):].strip()
        print("Джарвис:", reply)


if __name__ == "__main__":
    main()