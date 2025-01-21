import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
from torch.utils import data

from model import GPTLanguageModel

import numpy as np
import math

import wandb

import time
import os

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


class TinyStoriesDataset(data.IterableDataset):

    def __getitem__(self, index):
        pass

    def __init__(self, tokenized_path, block_size: int, device: str = 'cuda'):
        # Each line represents a short story
        self.block_size = block_size
        self.device = device
        self.train = np.load(tokenized_path, mmap_mode='r', allow_pickle=True)

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.train) - self.block_size, 1)[0]
            chunk = self.train[idx:idx + self.block_size + 1]
            source = torch.tensor(chunk[:-1], device=self.device, dtype=torch.long)
            target = torch.tensor(chunk[1:], device=self.device, dtype=torch.long)
            yield source, target

    def __len__(self):
        return len(self.train)


model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print(f"Tokens :{model.vocab_size}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=model.learning_rate)


# DATASET
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_TRAIN = 'data/TinyStories-train.txt'
TINY_TOKENIZED = 'data/tiny_tokenized.npy'


train_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED, model.block_size, device=model.device)
train_loader = data.DataLoader(train_tiny_stories, batch_size=model.batch_size)
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_VAL = 'data/TinyStories-valid.txt'
TINY_TOKENIZED_VAL = 'data/tiny_tokenized_val.npy'


val_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED_VAL, model.block_size, device=model.device)
val_loader = data.DataLoader(val_tiny_stories, batch_size=model.batch_size)

@torch.no_grad()
def estimate_loss():
    print("---------------------------------------------------------------")
    print("Mode: Evaluating")
    out = {}
    model.eval()
    losses = torch.zeros(model.eval_iters)
    for iter,batch in enumerate(val_loader):
        if iter == model.eval_iters:
            break
        X, Y = batch
        logits, loss = model(X, Y)
        losses[iter] = loss.item()
        print(f"Batch {iter}")
    model.train()
    print("---------------------------------------------------------------")
    print("Mode: Training")
    return losses.mean().item()

max_lr = 3e-4
min_lr = max_lr *0.1
warmup_steps = 250
max_steps = 50

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    if step >= max_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)






if os.path.exists(model.load_path):
    checkpoint = torch.load(model.load_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.init(
    project='TinyLM'
)

text_table = wandb.Table(columns=['epoch', 'loss', 'predicted text'])
losses=0

try:
    for iter,batch in enumerate(train_loader):
        t0 = time.time()
        # every once in a while evaluate the loss on train and val sets
        if (iter % model.eval_interval == 0)or iter == model.max_iters - 1:
            loss_eval = estimate_loss()
            print(f"step {iter}: val loss {loss_eval:.4f}")
            wandb.log({"val_loss": loss_eval})

        with torch.autocast(device_type=model.device, dtype=torch.bfloat16):   
            # sample a batch of data
            xb, yb = batch

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()    

        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        losses+=(loss.item())
        wandb.log({"lr": lr,"loss":loss.item()})
        if iter % 10 == 0:
            wandb.log({"loss(Avg)": (losses) / 10})
            losses = 0

        t1 = time.time()
        print(f"Batch {iter} | Time per Batch {t1-t0}")
        torch.cuda.empty_cache()

except KeyboardInterrupt:
    pass

finally:
    checkpoint_location = model.load_path

    print(f"Saving model to {checkpoint_location} and shutting down training...")
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_location)