# Transformer From Scratch

A complete, from-scratch PyTorch implementation of the encoder-decoder Transformer architecture introduced in **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** (Vaswani et al., 2017) — built as a learning project to understand the paper's mechanics by implementing them directly, without relying on `nn.Transformer` or any pre-built attention modules.

The model is trained on a small English → French translation task and includes a tool to visualize cross-attention weights, showing exactly which source words the decoder attends to while generating each target word.

---

## Overview

Most tutorials implement only the decoder (GPT-style) half of the Transformer. This project implements the **full architecture** described in the original paper:

- An **Encoder** stack that reads and contextualizes the source sentence
- A **Decoder** stack that generates the target sentence token by token
- **Cross-attention**, connecting the decoder to the encoder's output
- The paper's original **sinusoidal positional encoding** (fixed, not learned)

| Component | Implemented |
|---|---|
| Scaled Dot-Product Attention | ✅ |
| Multi-Head Attention | ✅ |
| Sinusoidal Positional Encoding | ✅ |
| Masked (causal) Decoder Self-Attention | ✅ |
| Encoder–Decoder Cross-Attention | ✅ |
| Position-wise Feed-Forward Networks | ✅ |
| Residual Connections + Post-LayerNorm | ✅ |
| Attention weight visualization | ✅ |

---

## Architecture

```
Source Sentence                          Target Sentence (shifted right)
      │                                           │
      ▼                                           ▼
Input Embedding + Positional Encoding    Output Embedding + Positional Encoding
      │                                           │
      ▼                                           ▼
┌─────────────────────┐                 ┌──────────────────────┐
│   ENCODER  × N       │                 │   DECODER  × N        │
│                      │                 │                       │
│ Multi-Head           │                 │ Masked Multi-Head     │
│ Self-Attention        │                 │ Self-Attention        │
│      + Add & Norm    │                 │      + Add & Norm     │
│                      │  ─────────────▶ │ Cross-Attention        │
│ Feed-Forward          │   Encoder       │ (Q: decoder,           │
│      + Add & Norm    │    Output       │  K,V: encoder output)  │
│                      │                 │      + Add & Norm     │
│                      │                 │                       │
│                      │                 │ Feed-Forward            │
│                      │                 │      + Add & Norm     │
└─────────────────────┘                 └──────────────────────┘
                                                    │
                                                    ▼
                                          Linear + Softmax
                                                    │
                                                    ▼
                                           Predicted next word
```

---

## Repository Structure

```
Transformer-From-Scratch/
├── model.py   # Core architecture: PositionalEncoding, MultiHeadAttention,
│              # EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer
├── main.py    # Toy dataset, training loop, greedy decoding, attention visualization
└── README.md
```

---

## Installation

```bash
git clone https://github.com/krishnadanp2057/Transformer-From-Scratch.git
cd Transformer-From-Scratch
pip install torch matplotlib
```

## Usage

```bash
python main.py
```

This will:
1. Build source/target vocabularies from a small toy English–French parallel corpus
2. Train the Transformer for 500 epochs (full-batch, since the dataset is small)
3. Print greedy-decoded translations for a few test sentences
4. Save an attention heatmap (`attention_heatmap.png`) showing cross-attention weights for a sample translation

Example output:

```
EN: attention is all you need
FR: l attention est tout ce dont vous avez besoin

EN: i love machine learning
FR: j aime l apprentissage automatique
```

## Attention Visualization

The `plot_attention()` function in `main.py` extracts the decoder's final-layer cross-attention weights during generation and renders them as a heatmap — source words on one axis, generated target words on the other. This provides a direct, visual way to confirm that the decoder is attending to semantically relevant source words (e.g. "learning" ↔ "apprentissage") rather than attending uniformly or at random.

---

## Implementation Notes

- **Tokenization** is word-level for simplicity. For a production system, this would be replaced with a subword tokenizer (BPE / WordPiece).
- **Positional encoding** uses the fixed sinusoidal formula from the paper rather than a learned embedding, matching Section 3.5 of the original work.
- **Normalization** follows the original paper's post-norm convention: `LayerNorm(x + Sublayer(x))`. Many modern implementations use pre-norm instead for training stability at larger scale.
- The training corpus is intentionally small (~16 sentence pairs) to keep the project runnable on CPU in under a minute. The architecture itself scales to larger datasets without modification.

## Possible Extensions

- [ ] Swap word-level tokenization for a trained BPE tokenizer
- [ ] Add a train/validation split and track validation loss
- [ ] Implement beam search decoding instead of greedy decoding
- [ ] Add dropout tuning and learning-rate warmup (as specified in the original paper)
- [ ] Scale to a larger parallel corpus (e.g. Multi30k or a Tatoeba subset)

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## Author

**Krishnandan Pandit**
[GitHub](https://github.com/krishnadanp2057)

## License

This project is open source and available under the [MIT License](LICENSE).
