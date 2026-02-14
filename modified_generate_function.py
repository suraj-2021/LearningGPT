def generate(model, idx, max_new_tokens, context_size, 
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        # Standard GPT forward pass
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]  # Last position
        
        # TOP-K: Safety guardrails
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < top_logits[:, -1], 
                               float('-inf'), logits)
        
        # TEMPERATURE: Creativity control
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)  # Random!
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy
        
        # Early stopping
        if idx_next == eos_id:
            break
            
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
