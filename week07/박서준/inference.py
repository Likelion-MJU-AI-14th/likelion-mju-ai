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


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def main():
	tokenizer = tiktoken.get_encoding("gpt2")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Device:", device)    

	BASE_CONFIG = {
		"vocab_size": 50257,     # Vocabulary size
		"context_length": 1024,  # Context length
		"drop_rate": 0.0,        # Dropout rate
		"qkv_bias": True         # Query-key-value bias
	}

	model_configs = {
		"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
		"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
		"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
		"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
	}

	CHOOSE_MODEL = "gpt2-medium (355M)"
	BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

	model = GPTModel(BASE_CONFIG)
	file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
	model.load_state_dict(torch.load(file_name, weights_only=True))
	model.eval()
	model.to(device)
 
	entry =  {
		"instruction": "Why is director Hong Myung-bo criticized?",
		"input": "",
	}
	input_text = format_input(entry)
	token_ids = generate(
		model=model,
		idx=text_to_token_ids(input_text, tokenizer).to(device),
		max_new_tokens=256,
		context_size=BASE_CONFIG["context_length"],
		eos_id=50256
	)
 
	generated_text = token_ids_to_text(token_ids, tokenizer)
	response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
	
	print("Response:", response_text)
 

if __name__ == "__main__":
    main()