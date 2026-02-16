!pip install tiktoken transformers tqdm torch

import numpy as np
import torch
from torch import nn
import tiktoken as ttk
from tqdm import tqdm
from dataclasses import dataclass
from transformers import GPT2Model

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = True

GPT_CONFIG_124M = GPTConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def assign_check(left, right, name):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch for {name}. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, dtype=torch.float32))

def load_weights_into_gpt(gpt_model, params):
    gpt_model.pos_emb.weight = assign_check(gpt_model.pos_emb.weight, params['wpe.weight'], 'wpe')
    gpt_model.tok_emb.weight = assign_check(gpt_model.tok_emb.weight, params['wte.weight'], 'wte')
    
    for b in tqdm(range(len(params['blocks'])), desc="Loading Transformer Blocks"):
        q_w, k_w, v_w = np.split(params['blocks'][b]['attn']['c_attn.weight'], 3, axis=-1)
        
        gpt_model.trf_blocks[b].att.w_queries.weight = assign_check(gpt_model.trf_blocks[b].att.w_queries.weight, q_w.T, 'q_w')
        gpt_model.trf_blocks[b].att.w_keys.weight    = assign_check(gpt_model.trf_blocks[b].att.w_keys.weight, k_w.T, 'k_w')
        gpt_model.trf_blocks[b].att.w_values.weight  = assign_check(gpt_model.trf_blocks[b].att.w_values.weight, v_w.T, 'v_w')

        q_b, k_b, v_b = np.split(params['blocks'][b]['attn']['c_attn.bias'], 3, axis=-1)
        gpt_model.trf_blocks[b].att.w_queries.bias = assign_check(gpt_model.trf_blocks[b].att.w_queries.bias, q_b, 'q_b')
        gpt_model.trf_blocks[b].att.w_keys.bias    = assign_check(gpt_model.trf_blocks[b].att.w_keys.bias, k_b, 'k_b')
        gpt_model.trf_blocks[b].att.w_values.bias  = assign_check(gpt_model.trf_blocks[b].att.w_values.bias, v_b, 'v_b')

        gpt_model.trf_blocks[b].att.out_proj.weight = assign_check(gpt_model.trf_blocks[b].att.out_proj.weight, params['blocks'][b]['attn']['c_proj.weight'].T, 'att_out_w')
        gpt_model.trf_blocks[b].att.out_proj.bias   = assign_check(gpt_model.trf_blocks[b].att.out_proj.bias, params['blocks'][b]['attn']['c_proj.bias'], 'att_out_b')

        gpt_model.trf_blocks[b].ff.layers[0].weight = assign_check(gpt_model.trf_blocks[b].ff.layers[0].weight, params['blocks'][b]['mlp']['c_fc.weight'].T, 'ff1_w')
        gpt_model.trf_blocks[b].ff.layers[0].bias   = assign_check(gpt_model.trf_blocks[b].ff.layers[0].bias, params['blocks'][b]['mlp']['c_fc.bias'], 'ff1_b')
        
        gpt_model.trf_blocks[b].ff.layers[2].weight = assign_check(gpt_model.trf_blocks[b].ff.layers[2].weight, params['blocks'][b]['mlp']['c_proj.weight'].T, 'ff2_w')
        gpt_model.trf_blocks[b].ff.layers[2].bias   = assign_check(gpt_model.trf_blocks[b].ff.layers[2].bias, params['blocks'][b]['mlp']['c_proj.bias'], 'ff2_b')

        gpt_model.trf_blocks[b].norm1.scale = assign_check(gpt_model.trf_blocks[b].norm1.scale, params['blocks'][b]['ln_1.weight'], 'ln1_s')
        gpt_model.trf_blocks[b].norm1.shift = assign_check(gpt_model.trf_blocks[b].norm1.shift, params['blocks'][b]['ln_1.bias'], 'ln1_b')
        gpt_model.trf_blocks[b].norm2.scale = assign_check(gpt_model.trf_blocks[b].norm2.scale, params['blocks'][b]['ln_2.weight'], 'ln2_s')
        gpt_model.trf_blocks[b].norm2.shift = assign_check(gpt_model.trf_blocks[b].norm2.shift, params['blocks'][b]['ln_2.bias'], 'ln2_b')

    gpt_model.final_norm.scale = assign_check(gpt_model.final_norm.scale, params['ln_f.weight'], 'ln_f_s')
    gpt_model.final_norm.shift = assign_check(gpt_model.final_norm.shift, params['ln_f.bias'], 'ln_f_b')
    gpt_model.out_head.weight = assign_check(gpt_model.out_head.weight, params['wte.weight'], 'out_head')
    
    print("-> Weights successfully loaded!")

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
model.eval()

hf_gpt2 = GPT2Model.from_pretrained("gpt2")
hf_sd = hf_gpt2.state_dict()

openai_params = {
    "wpe.weight": hf_sd["wpe.weight"].cpu().numpy(),
    "wte.weight": hf_sd["wte.weight"].cpu().numpy(),
    "ln_f.weight": hf_sd["ln_f.weight"].cpu().numpy(),
    "ln_f.bias": hf_sd["ln_f.bias"].cpu().numpy(),
    "blocks": []
}

for i in range(12):
    block_params = {
        "attn": {
            "c_attn.weight": hf_sd[f"h.{i}.attn.c_attn.weight"].cpu().numpy(),
            "c_attn.bias": hf_sd[f"h.{i}.attn.c_attn.bias"].cpu().numpy(),
            "c_proj.weight": hf_sd[f"h.{i}.attn.c_proj.weight"].cpu().numpy(),
            "c_proj.bias": hf_sd[f"h.{i}.attn.c_proj.bias"].cpu().numpy(),
        },
        "mlp": {
            "c_fc.weight": hf_sd[f"h.{i}.mlp.c_fc.weight"].cpu().numpy(),
            "c_fc.bias": hf_sd[f"h.{i}.mlp.c_fc.bias"].cpu().numpy(),
            "c_proj.weight": hf_sd[f"h.{i}.mlp.c_proj.weight"].cpu().numpy(),
            "c_proj.bias": hf_sd[f"h.{i}.mlp.c_proj.bias"].cpu().numpy(),
        },
        "ln_1.weight": hf_sd[f"h.{i}.ln_1.weight"].cpu().numpy(),
        "ln_1.bias": hf_sd[f"h.{i}.ln_1.bias"].cpu().numpy(),
        "ln_2.weight": hf_sd[f"h.{i}.ln_2.weight"].cpu().numpy(),
        "ln_2.bias": hf_sd[f"h.{i}.ln_2.bias"].cpu().numpy()
    }
    openai_params["blocks"].append(block_params)

load_weights_into_gpt(model, openai_params)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

tokenizer = ttk.get_encoding("gpt2")
start_context = "Every effort moves you"
print(f"\nPrompt: {start_context}")

encoded = text_to_token_ids(start_context, tokenizer).to(device)
generated_ids = generate_text_simple(model, encoded, max_new_tokens=25, context_size=GPT_CONFIG_124M.context_length)
generated_text = token_ids_to_text(generated_ids, tokenizer)

print("-" * 50)
print(f"Generated text:\n{generated_text}")
print("-" * 50)
