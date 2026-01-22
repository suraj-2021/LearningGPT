class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Embeddings (Chapters 1-2)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Stack of TransformerBlocks (this chapter!)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # Final processing
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):  # [batch, seq_len]
        # 1. Token + Position embeddings
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # 2. Transformer blocks (the magic happens here!)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        # 3. Output logits for next-token prediction
        logits = self.out_head(x)  # [batch, seq_len, vocab_size]
        return logits
