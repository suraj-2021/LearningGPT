import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import tiktoken as ttk
from tqdm import tqdm
import os
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 512
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        config_dict = self.to_dict()
        formatted_items = [f'"{key}": {repr(value)}' for key, value in config_dict.items()]
        return "GPT_CONFIG_124M = {\n " + ",\n ".join(formatted_items) + "\n}"

@dataclass
class DataConfig:
    dataPath: str = r'the-verdict.txt'
    max_length: int = GPTConfig.context_length
    batch_size: int = 64
    train_ratio: float = 0.90
    stride: int = GPTConfig.context_length

DataConfig = DataConfig()
GPTConfig = GPTConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        return raw_text
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

if not os.path.exists(DataConfig.dataPath):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    urllib.request.urlretrieve(url, DataConfig.dataPath)

raw_data = read_txt(DataConfig.dataPath)

tokenizer = ttk.get_encoding("gpt2")

total_token = len(tokenizer.encode(raw_data))
print(f"-> Number of Characters: {len(raw_data)}\n-> Number of Tokens: {total_token}")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer
        token_ids = tokenizer.encode(txt)
        self.input_ids = []
        self.target_ids = []

        for i in tqdm(range(0, len(token_ids) - max_length, stride)):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def LLM_DataLoader(txt, tokenizer, batch_size: int, max_length: int, stride: int,
                   shuffle: bool = True, drop_last: bool = True):
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

train_ratio = DataConfig.train_ratio
split_idx = int(train_ratio * len(raw_data))
train_data = raw_data[:split_idx]
val_data = raw_data[split_idx:]
print(f'-> Length of training data: {len(train_data)}\n-> Length of validation data: {len(val_data)}')

if total_token * train_ratio < GPTConfig.context_length:
    print("Not enough tokens for training. Lower context_length or increase train_ratio.")
if total_token * (1 - train_ratio) < GPTConfig.context_length:
    print("Not enough tokens for validation. Lower context_length or decrease train_ratio.")

train_dataloader = LLM_DataLoader(
    txt=train_data,
    tokenizer=tokenizer,
    max_length=DataConfig.max_length,
    batch_size=DataConfig.batch_size,
    stride=DataConfig.stride,
    shuffle=False,
    drop_last=False
)

val_dataloader = LLM_DataLoader(
    txt=val_data,
    tokenizer=tokenizer,
    max_length=DataConfig.max_length,
    batch_size=DataConfig.batch_size,
    stride=DataConfig.stride,
    shuffle=False,
    drop_last=False
)

dataiter = iter(train_dataloader)
firstbatch = next(dataiter)
print(f'Example inputs:\n{firstbatch[0]}\nExample targets:\n{firstbatch[1]}')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        batches, num_tokens, dim_in = x.shape

        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (queries @ keys.transpose(2, 3)) / (self.head_dim ** 0.5)

        attn_scores = attn_scores.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForwardGELU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg.emb_dim
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        resid_conn = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + resid_conn

        resid_conn = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + resid_conn
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

model = GPTModel(GPTConfig)
model.to(device)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

start_context = "Hello, I am"
encoded = text_to_token_ids(start_context, tokenizer).to(device)
generated_ids = generate_text_simple(model, encoded, max_new_tokens=10, context_size=GPTConfig.context_length)
generated_text = token_ids_to_text(generated_ids, tokenizer)
print(f"Generated text: {generated_text}")
