# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# A minimal instruction finetuning file based on the code in chapter 7

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time

import matplotlib.pyplot as plt
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from local files in this folder
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plot_name = "loss-plot-standalone.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main():
    #######################################
    # Print package versions
    #######################################
    print()
    pkgs = [
        "matplotlib",  # Plotting library
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
        "tqdm",        # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #######################################
    # Download and prepare dataset
    #######################################
    # 1. "instruction-data.json" 파일을 로드하기
    with open("instruction-data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"데이터셋 크기: {len(data)}")

    # 2. 데이터셋을 train/valid/test 세트로 분리하기
    train_portion = int(len(data) * 0.85)
    val_portion   = int(len(data) * 0.05)
    train_data = data[:train_portion]
    val_data   = data[train_portion:train_portion + val_portion]
    test_data  = data[train_portion + val_portion:]
    print(f"훈련: {len(train_data)}, 검증: {len(val_data)}, 테스트: {len(test_data)}")

    # 3. 각 세트에 대해 데이터로더 생성하기
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=True, drop_last=True, num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader  = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader  = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn,
        shuffle=False, drop_last=False, num_workers=num_workers
    )

    print(f"훈련 배치: {len(train_loader)}, 검증 배치: {len(val_loader)}, 테스트 배치: {len(test_loader)}")

    #######################################
    # Load pretrained model
    #######################################
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 1. 사전 학습된 "gpt2-medium (355M)" 모델 불러오기
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    # 2. 모델을 평가 모드로 전환 후 device로 이동
    model.eval()
    model.to(device)

    #######################################
    # Finetuning the model
    #######################################
    # 1. train_model_simple() 함수를 이용해서 모델을 미세 튜닝하기 (epoch는 2회)
    start_time = time.time()

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer
    )

    end_time = time.time()
    print(f"훈련 소요 시간: {(end_time - start_time) / 60:.2f}분")

    epochs_tensor      = torch.linspace(0, num_epochs, len(train_losses))
    tokens_seen_tensor = torch.linspace(0, tokens_seen[-1], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen_tensor, train_losses, val_losses)

    # 2. 미세 튜닝한 모델을 저장하기
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


if __name__ == "__main__":

    main()