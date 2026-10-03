"""
Trains the encoder-decoder Transformer (model.py) on a small toy
English -> French translation dataset, then:
  1. Translates a few sentences
  2. Visualizes cross-attention as a heatmap (saved as a PNG)

This is a TOY dataset (only ~16 sentence pairs) meant to demonstrate
the full paper architecture working end-to-end, not to produce
production-quality translation. With more data, the same code scales.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Transformer

# ============================================================
# 1. TOY PARALLEL CORPUS (English -> French)
# ============================================================
pairs = [
    ("hello", "bonjour"),
    ("good morning", "bon matin"),
    ("how are you", "comment allez vous"),
    ("thank you very much", "merci beaucoup"),
    ("see you tomorrow", "a demain"),
    ("what is your name", "quel est votre nom"),
    ("i love machine learning", "j aime l apprentissage automatique"),
    ("i am learning transformers", "j apprends les transformateurs"),
    ("this is a good project", "c est un bon projet"),
    ("artificial intelligence is powerful", "l intelligence artificielle est puissante"),
    ("the weather is nice today", "le temps est agreable aujourd hui"),
    ("i am a student", "je suis un etudiant"),
    ("this is my github project", "c est mon projet github"),
    ("deep learning is interesting", "l apprentissage profond est interessant"),
    ("attention is all you need", "l attention est tout ce dont vous avez besoin"),
    ("i am going to college", "je vais a l universite"),
]

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# ============================================================
# 2. BUILD VOCABULARIES (word-level, one per language)
# ============================================================
def build_vocab(sentences):
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


src_sentences = [p[0] for p in pairs]
tgt_sentences = [p[1] for p in pairs]

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
src_idx2word = {i: w for w, i in src_vocab.items()}
tgt_idx2word = {i: w for w, i in tgt_vocab.items()}

print(f"Source vocab size: {len(src_vocab)}")
print(f"Target vocab size: {len(tgt_vocab)}")


def encode(sentence, vocab, add_sos_eos=False):
    ids = [vocab.get(w, UNK_IDX) for w in sentence.split()]
    if add_sos_eos:
        ids = [SOS_IDX] + ids + [EOS_IDX]
    return ids


# ============================================================
# 3. PAD SEQUENCES TO FIXED LENGTH
# ============================================================
src_encoded = [encode(s, src_vocab) for s in src_sentences]
tgt_encoded = [encode(s, tgt_vocab, add_sos_eos=True) for s in tgt_sentences]

MAX_SRC_LEN = max(len(s) for s in src_encoded)
MAX_TGT_LEN = max(len(s) for s in tgt_encoded)


def pad(seq, max_len):
    return seq + [PAD_IDX] * (max_len - len(seq))


src_padded = torch.tensor([pad(s, MAX_SRC_LEN) for s in src_encoded], dtype=torch.long)
tgt_padded = torch.tensor([pad(s, MAX_TGT_LEN) for s in tgt_encoded], dtype=torch.long)

# Teacher forcing: decoder input = target shifted right (drop last token)
#                  decoder target = target shifted left (drop first token)
tgt_input = tgt_padded[:, :-1]
tgt_output = tgt_padded[:, 1:]

print(f"Max source length: {MAX_SRC_LEN}, Max target length: {MAX_TGT_LEN}")


# ============================================================
# 4. CREATE MODEL
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=64,
    num_heads=4,
    num_layers=2,
    d_ff=256,
    max_len=max(MAX_SRC_LEN, MAX_TGT_LEN) + 5,
    dropout=0.1,
    pad_idx=PAD_IDX,
).to(device)

src_padded = src_padded.to(device)
tgt_input = tgt_input.to(device)
tgt_output = tgt_output.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


# ============================================================
# 5. TRAINING LOOP (full-batch, since the dataset is tiny)
# ============================================================
EPOCHS = 500

for epoch in range(EPOCHS):
    model.train()
    logits, _ = model(src_padded, tgt_input)

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_output.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")


# ============================================================
# 6. GREEDY TRANSLATION (inference)
# ============================================================
def translate(sentence, max_len=15):
    model.eval()
    src_ids = encode(sentence, src_vocab)
    src_ids = pad(src_ids, MAX_SRC_LEN)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

    src_mask = model.make_src_mask(src_tensor)
    enc_out = model.encoder(src_tensor, src_mask)

    tgt_ids = [SOS_IDX]
    cross_attn_weights = None

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        logits, cross_attn_weights = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
        next_token = logits[0, -1].argmax().item()
        tgt_ids.append(next_token)

        if next_token == EOS_IDX:
            break

    words = [tgt_idx2word[i] for i in tgt_ids[1:] if i not in (EOS_IDX, PAD_IDX)]
    return " ".join(words), src_ids, tgt_ids, cross_attn_weights


# Try a few translations
print("\n--- Translations ---")
test_sentences = [
    "hello",
    "i love machine learning",
    "attention is all you need",
    "this is my github project",
]
for s in test_sentences:
    translation, _, _, _ = translate(s)
    print(f"EN: {s}")
    print(f"FR: {translation}\n")


# ============================================================
# 7. VISUALIZE CROSS-ATTENTION AS A HEATMAP
# ============================================================
def plot_attention(sentence, save_path="attention_heatmap.png"):
    translation, src_ids, tgt_ids, cross_attn = translate(sentence)

    # cross_attn shape: [batch=1, heads, tgt_len, src_len]
    # Average across heads for a single clean heatmap.
    attn = cross_attn[0].mean(dim=0).detach().cpu().numpy()  # [tgt_len, src_len]

    src_tokens = [src_idx2word[i] for i in src_ids if i != PAD_IDX]
    tgt_tokens = [tgt_idx2word[i] for i in tgt_ids[1:] if i not in (EOS_IDX, PAD_IDX)]

    # Trim attention matrix to actual (non-padded) lengths
    attn = attn[: len(tgt_tokens), : len(src_tokens)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="viridis")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("Source sentence (English)")
    ax.set_ylabel("Generated sentence (French)")
    ax.set_title(f'Cross-Attention: "{sentence}" -> "{translation}"', fontsize=11, wrap=True, pad=15)

    fig.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention heatmap to {save_path}")


plot_attention("i love machine learning")
