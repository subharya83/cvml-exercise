"""
tinyvlm_model.py
-----------------
The architecture shared by trainVLM.py and inferVLM.py:

    image --[frozen CLIP]--> pooled embedding --[projector]--> 1 "visual token"
    visual token + caption tokens --[tiny Llama-style decoder]--> next-token logits

The decoder is a from-scratch, from-scratch-TRAINED reimplementation of
Andrej Karpathy's llama2.c architecture (RMSNorm, rotary position
embeddings, SwiGLU, no biases, tied input/output embeddings). Its weights
train end to end on (image, caption) pairs, random-initialized by default.
Loading a real llama2.c checkpoint (e.g. stories15M.bin, trained on
TinyStories text) is wired up as an explicit, opt-in comparison -- see
`load_llama2c_checkpoint()` and the README section "A note on stories15M" --
because that checkpoint's language prior is real but was never trained to
condition on a visual prefix token, so whether it helps here is a measured
question, not an assumption.

Shapes referenced throughout (B=batch, T=caption length incl. BOS/EOS,
D=decoder.dim, V=vocab size):

    pixel_values     [B, 3, 224, 224]
    clip embedding   [B, clip_dim]            (clip_dim=512 for ViT-B/32)
    visual token     [B, 1, D]
    caption_ids      [B, T]                    (includes BOS at 0, EOS at end)
    decoder input x  [B, T, D]  = concat(visual_token, embed(caption_ids[:, :-1]))
    logits           [B, T, V]
    training target  caption_ids[:, 1:]         -> [B, T-1]
    training logits  logits[:, 1:, :]           -> [B, T-1, V]   (see forward())
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class DecoderConfig:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6      # <= n_heads; n_heads % n_kv_heads must be 0 (GQA)
    vocab_size: int = 0      # always set at runtime from the active tokenizer
    max_seq_len: int = 256
    multiple_of: int = 32
    dropout: float = 0.0


# ----------------------------------------------------------------------------
# Building blocks (RMSNorm, RoPE, SwiGLU) -- the three ingredients that turn a
# "vanilla" transformer decoder into a Llama-family one.
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention (n_kv_heads <=
    n_heads): each group of (n_heads // n_kv_heads) query heads shares one
    key/value head, repeated to match. With n_kv_heads == n_heads (this
    build's default) it collapses to ordinary multi-head attention.

    The visual token is just the first element of the sequence, attended to
    causally like any other position -- this is visual-PREFIX self-attention,
    not cross-attention (there are no separate image keys/values; queries,
    keys, and values all come from the same one sequence)."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        hidden = int(2 * (4 * cfg.dim) / 3)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLlama(nn.Module):
    """A minimal, from-scratch reimplementation of the llama2.c model.py
    architecture -- small enough to read in one sitting, and weight-layout
    compatible with karpathy/llama2.c legacy .bin checkpoints. Input and
    output embeddings are tied (as in llama2.c's "shared classifier" mode),
    which both matches the reference format and keeps the parameter count
    honest for a from-scratch build."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "vocab_size must be set from the active tokenizer"
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # tied embeddings
        cos, sin = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, tokens_or_embeds: torch.Tensor, is_embeds: bool = False):
        if is_embeds:
            x = tokens_or_embeds
            assert x.dim() == 3 and x.shape[-1] == self.cfg.dim, \
                f"expected embeddings [B, T, {self.cfg.dim}], got {tuple(x.shape)}"
        else:
            x = self.tok_embeddings(tokens_or_embeds)
        T = x.shape[1]
        assert T <= self.cfg.max_seq_len, \
            f"sequence length {T} exceeds decoder.max_seq_len={self.cfg.max_seq_len}"
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.output(x)


# ----------------------------------------------------------------------------
# Visual projector: 1 CLIP embedding -> 1 decoder "visual token"
# ----------------------------------------------------------------------------
class VisualProjector(nn.Module):
    def __init__(self, clip_dim: int, decoder_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, decoder_dim),
        )

    def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
        assert clip_embed.dim() == 2, f"expected [B, clip_dim], got {tuple(clip_embed.shape)}"
        return self.net(clip_embed).unsqueeze(1)  # (B, 1, decoder_dim)


# ----------------------------------------------------------------------------
# llama2.c legacy checkpoint loader (weights only -- see tokenizer notes in
# README / trainVLM.py for the paired tokenizer.model file)
#   Format (see https://github.com/karpathy/llama2.c export.py, version 0):
#   header = 7 x int32 (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
#                        vocab_size, max_seq_len)
#   followed by float32 arrays in a fixed order for every weight tensor.
# ----------------------------------------------------------------------------
def load_llama2c_checkpoint(path: str, model: TinyLlama) -> dict:
    """Best-effort loader for a llama2.c legacy .bin file (e.g. stories15M.bin).
    Copies matching weights into `model` in place and returns the raw header
    fields so the caller can sanity-check shapes against the config in use.
    This loads WEIGHTS ONLY -- pair it with the real tokenizer.model (a
    standard SentencePiece file) built from the same vocabulary, not with
    llama2.c's tokenizer.bin, which is a different, C-runtime-only export
    format that a SentencePieceProcessor cannot open directly."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with open(p, "rb") as f:
        header = struct.unpack("7i", f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = header
        shared_classifier = True
        if vocab_size < 0:
            vocab_size = -vocab_size
            shared_classifier = False

        def read(n):
            return torch.from_numpy(
                __import__("numpy").fromfile(f, dtype="float32", count=n)
            )

        tok_emb = read(vocab_size * dim).view(vocab_size, dim)
        attn_norm = [read(dim) for _ in range(n_layers)]
        wq = [read(dim * dim).view(dim, dim) for _ in range(n_layers)]
        wk = [read(dim * dim).view(dim, dim) for _ in range(n_layers)]
        wv = [read(dim * dim).view(dim, dim) for _ in range(n_layers)]
        wo = [read(dim * dim).view(dim, dim) for _ in range(n_layers)]
        ffn_norm = [read(dim) for _ in range(n_layers)]
        w1 = [read(hidden_dim * dim).view(hidden_dim, dim) for _ in range(n_layers)]
        w2 = [read(dim * hidden_dim).view(dim, hidden_dim) for _ in range(n_layers)]
        w3 = [read(hidden_dim * dim).view(hidden_dim, dim) for _ in range(n_layers)]
        final_norm = read(dim)

    sd = model.state_dict()
    sd["tok_embeddings.weight"].copy_(tok_emb[: sd["tok_embeddings.weight"].shape[0]])
    for i in range(min(n_layers, len(model.layers))):
        sd[f"layers.{i}.attn_norm.weight"].copy_(attn_norm[i])
        sd[f"layers.{i}.attn.wq.weight"].copy_(wq[i])
        sd[f"layers.{i}.attn.wk.weight"].copy_(wk[i])
        sd[f"layers.{i}.attn.wv.weight"].copy_(wv[i])
        sd[f"layers.{i}.attn.wo.weight"].copy_(wo[i])
        sd[f"layers.{i}.ffn_norm.weight"].copy_(ffn_norm[i])
        sd[f"layers.{i}.ffn.w1.weight"].copy_(w1[i])
        sd[f"layers.{i}.ffn.w2.weight"].copy_(w2[i])
        sd[f"layers.{i}.ffn.w3.weight"].copy_(w3[i])
    sd["norm.weight"].copy_(final_norm)
    if shared_classifier:
        sd["output.weight"].copy_(tok_emb[: sd["output.weight"].shape[0]])
    model.load_state_dict(sd)
    return dict(
        dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
        n_kv_heads=n_kv_heads, vocab_size=vocab_size, max_seq_len=max_seq_len,
    )


# ----------------------------------------------------------------------------
# Fallback tokenizer: character-level, built from the training captions.
# This is the DEFAULT tokenizer for the whole pipeline -- zero external
# downloads needed to run end to end.
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.specials = ["<pad>", "<bos>", "<eos>"]
        self.itos = self.specials + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.bos_id] + [self.stoi.get(c, self.pad_id) for c in text]
        ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i] if i < len(self.itos) else "?")
        return "".join(out)


# ----------------------------------------------------------------------------
# The full VLM: frozen CLIP -> projector -> TinyLlama decoder
# ----------------------------------------------------------------------------
class TinyVLM(nn.Module):
    def __init__(self, decoder: TinyLlama, projector: VisualProjector, clip_model):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.projector = projector
        self.decoder = decoder

    def train(self, mode: bool = True):
        """Override so `model.train()` (called every step by the training
        loop) can never accidentally flip the frozen CLIP encoder into
        train mode -- that would turn on dropout / let BatchNorm-style
        running stats drift inside a module we never backprop into anyway."""
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip.get_image_features(pixel_values=pixel_values)
        return out  # (B, clip_dim), pooled embedding

    def forward(self, pixel_values: torch.Tensor, caption_ids: torch.Tensor,
                clip_features: torch.Tensor | None = None):
        """Teacher-forced training forward pass.

        Sequence fed to the decoder:  [visual, BOS, tok_1, ..., tok_{T-2}]  (length T)
        Target (next-token) sequence:      caption_ids[:, 1:]  =  [tok_1, ..., tok_{T-1}]  (length T-1)

        Decoder output position i predicts input position i+1: the BOS
        position (index 1) attends back to the preceding visual token and
        predicts tok_1; the tok_1 position predicts tok_2; and so on. The
        logits at the visual-token position (index 0) predict BOS, which
        isn't part of the target we score -- they're excluded from the
        loss entirely, which is exactly what slicing off index 0 does.
        So the logits that align with `caption_ids[:, 1:]` are
        `logits[:, 1:, :]`, NOT `logits[:, :-1, :]` (that earlier,
        incorrect slice trained every position against its predecessor's
        label instead of its own next token, silently shifting supervision
        by one position -- loss still went down, but generation was never
        taught the right thing).

        `clip_features`, if given, are precomputed CLIP embeddings (see the
        feature cache in tinyvlm_data.py) -- skips redundant CLIP forward
        passes for the same image across its different captions/epochs.
        """
        img_feat = clip_features if clip_features is not None else self.encode_image(pixel_values)
        visual_tok = self.projector(img_feat)  # (B, 1, dim)
        tok_embeds = self.decoder.tok_embeddings(caption_ids[:, :-1])
        x = torch.cat([visual_tok, tok_embeds], dim=1)  # (B, T, dim)
        logits = self.decoder(x, is_embeds=True)        # (B, T, vocab)
        return logits[:, 1:, :]  # aligned with caption_ids[:, 1:]

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, bos_id: int, eos_id: int,
                 max_new_tokens: int = 40, temperature: float = 0.8, top_k: int = 40,
                 greedy: bool = True, clip_features: torch.Tensor | None = None,
                 generator: torch.Generator | None = None):
        self.eval()
        img_feat = clip_features if clip_features is not None else self.encode_image(pixel_values)
        visual_tok = self.projector(img_feat)
        cur = torch.tensor([[bos_id]], device=pixel_values.device if pixel_values is not None else img_feat.device)
        embeds = torch.cat([visual_tok, self.decoder.tok_embeddings(cur)], dim=1)
        generated = [bos_id]
        max_len = self.decoder.cfg.max_seq_len
        for _ in range(max_new_tokens):
            if embeds.shape[1] >= max_len:
                break  # respect the decoder's trained context length
            logits = self.decoder(embeds, is_embeds=True)
            next_logits = logits[:, -1, :]
            if greedy:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=generator)
            generated.append(next_id.item())
            if next_id.item() == eos_id:
                break
            next_embed = self.decoder.tok_embeddings(next_id)
            embeds = torch.cat([embeds, next_embed], dim=1)
        return generated
