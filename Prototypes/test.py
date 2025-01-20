import os.path
from datetime import datetime

import lora
import wandb

import torch
from tqdm import tqdm

import model
from transformers import GPT2TokenizerFast
from torch.utils import data
import numpy as np


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

config = {
    "BLOCK_SIZE": 128,
    "EMB_SIZE": 768,
    "N_ATTENTION_HEADS": 4,
    "N_DECODER_BLOCKS": 2,
    "VOCAB_SIZE": len(tokenizer),
    "MAX_OUT_TOKENS": 200,
    "EVAL_INTERVAL": 1000,
    "EVAL_ITER": 100,
    "LR": 3e-4,
    "BATCH_SIZE": 64 ,
    "DEVICE":'cpu',
    "LOAD_PATH": 'models/tiny_base.pt',
    "SAVE_PATH": 'models/tiny_base.pt',
    "ENABLE_LORA": True,
}

model = model.TinyLM(emb_dim=config['EMB_SIZE'], block_size=config['BLOCK_SIZE'],
                     n_att_heads=config['N_ATTENTION_HEADS'], n_decoders=config['N_DECODER_BLOCKS'],
                     vocab_size=config['VOCAB_SIZE'], device=config['DEVICE'])

checkpoint = torch.load(config['LOAD_PATH'], map_location=config['DEVICE'])
model.load_state_dict(checkpoint['model_state_dict'],strict=False)


@torch.no_grad()
def generate_sample_text(training_model, max_tokens: int = 200) -> str:
    training_model.eval()
    context = torch.zeros((5, config['BLOCK_SIZE']), dtype=torch.long, device=config['DEVICE'])
    out_tokens = training_model.generate(context, max_new_tokens=max_tokens)
    # Reform to one long piece of text
    out_tokens = out_tokens.view(out_tokens.shape[0] * out_tokens.shape[1])
    training_model.train()
    return tokenizer.decode(out_tokens)

print(generate_sample_text(model, 200))