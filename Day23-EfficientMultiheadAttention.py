class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        assert d_out % num_heads == 0
        self.num_heads, self.head_dim = num_heads, d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # NEW!
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):  # [b, seq, embed]
        # Single QKV → [b, seq, d_out]
        Q = self.W_query(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(b, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Parallel attn [b, heads, seq, seq]
        scores = Q @ K.transpose(-2, -1)
        scores.masked_fill_(mask[:seq, :seq], -torch.inf)
        weights = F.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        weights = self.dropout(weights)
        
        # Merge: [b, seq, d_out]
        out = (weights @ V).transpose(1, 2).contiguous().view(b, seq, self.d_out)
        return self.out_proj(out)
