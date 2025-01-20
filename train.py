import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2TokenizerFast
from torch.utils import data

import numpy as np
import math

import wandb

import time
import os

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
vocab_size = len(tokenizer) # how many unique tokens exist in our model's vocab?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 768
n_head = 4
n_layer = 2
dropout = 0.2
# ------------

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
    "BATCH_SIZE": 64,
    "DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu',
    "LOAD_PATH": 'models/tiny_base.pt',
    "ENABLE_LORA": True,
}

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


# DATASET
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_TRAIN = 'data/TinyStories-train.txt'
TINY_TOKENIZED = 'data/tiny_tokenized.npy'


train_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED, config['BLOCK_SIZE'], device=config['DEVICE'])
train_loader = data.DataLoader(train_tiny_stories, batch_size=config['BATCH_SIZE'])
# ----------------------------------------------------------------------------------------------------------------------
TINY_STORY_VAL = 'data/TinyStories-valid.txt'
TINY_TOKENIZED_VAL = 'data/tiny_tokenized_val.npy'


val_tiny_stories = TinyStoriesDataset(TINY_TOKENIZED_VAL, config['BLOCK_SIZE'], device=config['DEVICE'])
val_loader = data.DataLoader(val_tiny_stories, batch_size=config['BATCH_SIZE'])



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for iter,batch in enumerate(train_loader):
            if iter == eval_iters:
                break
            X, Y = batch
            logits, loss = model(X, Y)
            losses[iter] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

max_lr = 3e-4
min_lr = max_lr *0.1
warmup_steps = 10
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


model = GPTLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if os.path.exists(config['LOAD_PATH']):
    checkpoint = torch.load(config['LOAD_PATH'], map_location=config['DEVICE'])
    prev_epochs = checkpoint['epoch']
    lora_enabled_on_base = checkpoint['lora_was_enabled']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.init(
    project='TinyLM',
    config=config
)

text_table = wandb.Table(columns=['epoch', 'loss', 'predicted text'])
losses=0

try:
    for iter,batch in enumerate(train_loader):
        t0 = time.time()
        # every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval == 0 and iter!=0)or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            wandb.log({"val_loss": losses['val']})

        with torch.autocast(device_type=config['DEVICE'],dtype=torch.bfloat16):   
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
        print(f"Batch {iter} | Time per Batch {t1-t0}",end='\r')

except KeyboardInterrupt:
    pass

finally:
    checkpoint_location = config['LOAD_PATH']

    print(f"Saving model to {checkpoint_location} and shutting down training...")
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lora_was_enabled': config['ENABLE_LORA'],
                'config': config,
                }, checkpoint_location)