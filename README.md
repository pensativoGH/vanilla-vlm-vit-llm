# vanilla-vit-gpt

From-scratch PyTorch implementations of a Vision Transformer (ViT) for image classification and a GPT for character-level language modeling. Both models share the same hand-rolled `MultiHeadAttention`, pre-norm `TransformerBlock`, and custom `LayerNormalization` — the only meaningful difference is ViT uses non-causal attention with a CLS token, and GPT uses causal attention with token/position embeddings.

## Notebooks

- `ViT.ipynb` — Vision Transformer trained on CIFAR-10
- `GPT.ipynb` — character-level GPT trained on Tiny Shakespeare

## Data

### CIFAR-10 (ViT)

Source: https://github.com/EN10/CIFAR

The notebook expects `./cifar10/{train,test}_{images,labels}.npy` — the `.npy` dumps produced by that repo. Images are 32×32 RGB across 10 classes. Pixels are scaled to `[0, 1]` and normalized with the standard per-channel stats (mean `[0.4914, 0.4822, 0.4465]`, std `[0.2470, 0.2435, 0.2616]`), then transposed to NCHW.

### Tiny Shakespeare (GPT)

Source: https://github.com/karpathy/char-rnn (`data/tinyshakespeare/input.txt`)

Single-file concatenation of Shakespeare's works, ~1.1M characters. A simple character-level tokenizer builds the vocabulary from `sorted(set(text))` — no BPE, no special tokens. 90/10 train/val split on the token stream. The notebook expects the file at `./data/tinyshakespeare/input.txt`.

## Model & Training Details

Shared architecture choices:
- Pre-norm transformer blocks: `x + MHSA(LN(x))` then `x + FFN(LN(x))`
- FFN: `Linear(d, 4d) → GELU → Linear(4d, d)`, no bias
- Q/K/V/O projections: no bias
- Custom `LayerNormalization` (not `nn.LayerNorm`), eps `1e-5`
- Optimizer: AdamW, lr `3e-4`, betas `(0.9, 0.95)`, eps `1e-8`
- Device: Apple MPS

### ViT (CIFAR-10)

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Patch size | 8×8 |
| Patches per image | 16 (4×4 grid on 32×32) |
| Patch dim | 192 (3·8·8) |
| Classes | 10 |
| Batch size | 64 |
| Training steps | ~15,620 (≈20 epochs over 781 batches/epoch) |
| Attention | Non-causal |

Patches are formed via `unfold` on H and W, flattened to `(B, T, C·P·P)`, and linearly projected to `model_dim`. A learnable CLS token is prepended, and a learnable position embedding of length `num_patches + 1` is added. Classification uses the final CLS hidden state through a linear head.

Result on held-out CIFAR-10 test set: **val loss 1.206, val accuracy 56.7%**.

### GPT (Tiny Shakespeare)

| Hyperparameter | Value |
|---|---|
| Embedding dim | 64 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Context length (`block_size`) | 32 |
| Vocab size | Character-level (derived from data, ~65) |
| Batch size | 4 |
| Training steps | 200,000 |
| Attention | Causal (lower-triangular mask) |

Token and learnable position embeddings are summed, passed through the stack, and projected to vocab logits. Loss is cross-entropy over all positions, `(B·T, V)` vs. `(B·T,)`. Targets are the input shifted by one (standard next-token objective).

Final training loss settles around ~1.7–2.0 (noisy due to the tiny batch size of 4).

## Running

Each notebook is self-contained. Open in Jupyter and run top-to-bottom. The data paths (`./cifar10/` for ViT, `./data/tinyshakespeare/input.txt` for GPT) are hardcoded — drop the files in place before training.
