"""
Full Encoder-Decoder Transformer, implemented from scratch,
following "Attention Is All You Need" (Vaswani et al., 2017).

Unlike a decoder-only GPT, this has:
  - An Encoder stack (reads the source sentence, e.g. English)
  - A Decoder stack (generates the target sentence, e.g. French)
  - Cross-attention: the decoder attends over the encoder's output
  - Sinusoidal positional encoding (fixed, not learned)
"""

import math
import torch
import torch.nn as nn


# ============================================================
# 1. SINUSOIDAL POSITIONAL ENCODING (paper's exact formula)
# ============================================================
class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Unlike a learned nn.Embedding for position, this is FIXED —
    computed once with sin/cos and never updated during training.
    This is exactly what the original paper uses.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 2. MULTI-HEAD ATTENTION (used for self-attn AND cross-attn)
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    General-purpose multi-head attention.
    Used three ways in this model:
      - Encoder self-attention   (query=key=value=encoder input)
      - Decoder self-attention   (query=key=value=decoder input, causal-masked)
      - Decoder cross-attention  (query=decoder, key=value=encoder output)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, h, T, d_k]

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        Q = self.split_heads(self.w_q(query))
        K = self.split_heads(self.w_k(key))
        V = self.split_heads(self.w_v(value))

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, h, Tq, Tk]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # [B, h, Tq, Tk]
        out = attn_weights @ V  # [B, h, Tq, d_k]

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.w_o(out)

        return out, attn_weights  # weights returned for visualization later


# ============================================================
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. ENCODER LAYER
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Post-norm, exactly as in the original paper:
        # LayerNorm(x + Sublayer(x))
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ============================================================
# 5. DECODER LAYER
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked self-attention (causal — can't see future target tokens)
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Cross-attention — decoder queries attend over encoder output.
        #    THIS is what gets visualized: "which source words is the
        #    decoder looking at while generating this target word?"
        cross_out, cross_attn_weights = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x, cross_attn_weights


# ============================================================
# 6. ENCODER STACK
# ============================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ============================================================
# 7. DECODER STACK
# ============================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        cross_attn_weights = None
        for layer in self.layers:
            x, cross_attn_weights = layer(x, enc_out, src_mask, tgt_mask)

        logits = self.fc_out(x)
        # Return the LAST layer's cross-attention weights for visualization
        return logits, cross_attn_weights


# ============================================================
# 8. FULL SEQ2SEQ TRANSFORMER
# ============================================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64,
                 num_heads=4, num_layers=2, d_ff=256, max_len=50, dropout=0.1,
                 pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        # [B, 1, 1, src_len] — 0 where padding, 1 elsewhere
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        causal_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool()  # [T,T]
        return pad_mask & causal_mask  # broadcasts to [B,1,T,T]

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        logits, cross_attn = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return logits, cross_attn
