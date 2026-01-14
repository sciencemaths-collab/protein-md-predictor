from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_levels: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_levels = int(n_levels)
        self.d_model = int(d_model)
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(max_len, self.d_model)
        self.lvl_emb = nn.Embedding(self.n_levels, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                               dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        self.max_len = int(max_len)

    def forward(self, x_tok: torch.Tensor, level_id: torch.Tensor) -> torch.Tensor:
        B, C = x_tok.shape
        if C > self.max_len:
            raise ValueError(f"context length {C} exceeds max_len {self.max_len}")
        pos = torch.arange(C, device=x_tok.device).unsqueeze(0).expand(B, C)
        h = self.tok_emb(x_tok) + self.pos_emb(pos) + self.lvl_emb(level_id).unsqueeze(1)
        attn_mask = torch.triu(torch.ones(C, C, device=x_tok.device), diagonal=1).bool()
        h = self.encoder(h, mask=attn_mask)
        h = self.ln(h[:, -1, :])
        return self.head(h)

def energy_biased_logits(logits: torch.Tensor, E: torch.Tensor, beta: float) -> torch.Tensor:
    return logits - beta * E.unsqueeze(0)

def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)

def dist_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
