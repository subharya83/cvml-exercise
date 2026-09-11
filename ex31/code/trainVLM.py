#!/usr/bin/env python3
"""
trainVLM.py -- train a tiny Visual Language Model on a laptop.

    image --[frozen CLIP encoder]--> embedding --[trainable projector]-->
        1 visual token --[tiny Llama-style decoder]--> caption

Designed to be started from the command line at the START of a lecture and
left running quietly in a terminal while the slides cover why each piece
exists; by the "inference" slide there should be a few checkpoints on disk.

Usage:
    python trainVLM.py --config config.demo.yaml     # tested, ~350-step lecture run
    python trainVLM.py --config config.full.yaml      # a longer, non-lecture run
    python trainVLM.py --config config.demo.yaml --resume checkpoints/last.pt
    python trainVLM.py --config config.demo.yaml --smoke-test   # 30-second sanity check

Run `python trainVLM.py --help` for the full list of overridable flags.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import yaml

from tinyvlm_data import (CaptionDataset, ClipFeatureCache, export_demo_holdout,
                           load_flickr8k_split, make_collate)
from tinyvlm_model import (CharTokenizer, DecoderConfig, TinyLlama, TinyVLM,
                            VisualProjector, load_llama2c_checkpoint)


# ----------------------------------------------------------------------------
# CLI / config plumbing
# ----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="config.demo.yaml",
                   help="path to the YAML config (default: config.demo.yaml)")
    p.add_argument("--data-dir", type=str, default=None, help="override data.cache_dir")
    p.add_argument("--out-dir", type=str, default=None, help="override train.out_dir")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--max-iters", type=int, default=None,
                   help="stop after N optimizer steps (0 = full epochs); handy for demos")
    p.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--freeze-decoder", dest="freeze_decoder", action="store_true", default=None,
                   help="freeze the decoder, train only the projector -- only sensible once "
                        "the decoder already has a meaningful language prior (text-pretrained "
                        "or multimodal); with the default random-init decoder, freezing it "
                        "prevents the model from learning language at all")
    p.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from "
                   "(restores weights, optimizer state, step count, and as much RNG state as "
                   "is practical -- see README for exactly what is and isn't covered)")
    p.add_argument("--seed", type=int, default=None, help="override data.seed")
    p.add_argument("--smoke-test", action="store_true",
                   help="overfit 8 training examples for a few hundred steps as a fast "
                        "correctness check -- if loss doesn't approach ~0, something (e.g. the "
                        "teacher-forcing alignment) is broken, before you spend real time training")
    return p


def load_config(path: str, args: argparse.Namespace) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if args.data_dir: cfg["data"]["cache_dir"] = args.data_dir
    if args.out_dir: cfg["train"]["out_dir"] = args.out_dir
    if args.epochs is not None: cfg["train"]["epochs"] = args.epochs
    if args.max_iters is not None: cfg["train"]["max_iters"] = args.max_iters
    if args.batch_size is not None: cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None: cfg["train"]["lr"] = args.lr
    if args.device is not None: cfg["train"]["device"] = args.device
    if args.freeze_decoder is not None: cfg["train"]["freeze_decoder"] = args.freeze_decoder
    if args.seed is not None: cfg["data"]["seed"] = args.seed
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_rng_snapshot(train_ds) -> dict:
    """Best-effort RNG snapshot for --resume. Covers Python's `random`
    (which drives CaptionDataset's per-epoch caption sampling and the
    default DataLoader shuffle order, since neither is given its own
    generator), NumPy, and PyTorch's CPU and CUDA generators. This is
    "as reproducible as a single-process, num_workers=0-ish run gets" --
    NOT a guarantee of bit-for-bit resume under multi-worker DataLoader
    reshuffling, which has its own per-worker seeding. Treat --resume as
    "continues training sensibly," not "replays the exact same sequence.\""""
    import numpy as np
    snap = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    base_ds = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
    if hasattr(base_ds, "rng"):
        snap["caption_rng"] = base_ds.rng.getstate()
    return snap


def restore_rng_snapshot(snap: dict, train_ds):
    import numpy as np
    if "python_random" in snap:
        random.setstate(snap["python_random"])
    if "numpy_random" in snap:
        np.random.set_state(snap["numpy_random"])
    if "torch_cpu" in snap:
        torch.set_rng_state(snap["torch_cpu"])
    if "torch_cuda" in snap and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["torch_cuda"])
    base_ds = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
    if "caption_rng" in snap and hasattr(base_ds, "rng"):
        base_ds.rng.setstate(snap["caption_rng"])


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------
# Model assembly: CLIP (frozen) + tiny Llama decoder (random init by default;
# see README "A note on stories15M" for the opt-in pretrained-text experiment)
# + a small trainable projector between them.
# ----------------------------------------------------------------------------
def build_model(cfg: dict, tokenizer_vocab_size: int, device: torch.device) -> TinyVLM:
    from transformers import CLIPModel

    print(f"[model] loading frozen vision encoder: {cfg['vision_encoder']['name']}")
    clip_model = CLIPModel.from_pretrained(cfg["vision_encoder"]["name"]).to(device)

    dcfg = DecoderConfig(
        dim=cfg["decoder"]["dim"], n_layers=cfg["decoder"]["n_layers"],
        n_heads=cfg["decoder"]["n_heads"], n_kv_heads=cfg["decoder"]["n_kv_heads"],
        vocab_size=tokenizer_vocab_size, max_seq_len=cfg["decoder"]["max_seq_len"],
        multiple_of=cfg["decoder"]["multiple_of"], dropout=cfg["decoder"]["dropout"],
    )
    decoder = TinyLlama(dcfg)

    ckpt_path = cfg["decoder"].get("init_checkpoint")
    if ckpt_path and Path(ckpt_path).exists() and tokenizer_vocab_size >= 32000:
        # Opt-in comparison, off by default. stories15M was trained purely on
        # TinyStories TEXT -- it has a real English prior (grammar,
        # vocabulary, common next-token statistics) but no notion of
        # conditioning on a visual prefix token, since it never saw one.
        # Whether that prior helps captioning enough to be worth the vocab/
        # tokenizer coupling it requires is a measured question, not an
        # assumption -- that's what this path is for.
        try:
            info = load_llama2c_checkpoint(ckpt_path, decoder)
            print(f"[model] EXPERIMENT: seeded decoder from a text-only llama2.c "
                  f"checkpoint: {ckpt_path} (dim={info['dim']}, n_layers={info['n_layers']})")
            print("        it has a real language prior from TinyStories text, but no visual "
                  "grounding -- compare its loss curve against a random-init run yourself.")
        except Exception as e:
            print(f"[model] WARNING: could not load {ckpt_path} ({e}); using random init")
    else:
        print("[model] decoder starts from random init, trained end to end on "
              "(image, caption) pairs -- the default, tested path.")

    projector = VisualProjector(
        clip_dim=clip_model.config.projection_dim,
        decoder_dim=dcfg.dim,
        hidden_dim=cfg["projector"]["hidden_dim"],
    )

    model = TinyVLM(decoder=decoder, projector=projector, clip_model=clip_model).to(device)
    if cfg["train"]["freeze_decoder"]:
        for p in model.decoder.parameters():
            p.requires_grad = False
        if not (ckpt_path and Path(ckpt_path).exists()):
            print("[model] WARNING: decoder frozen but no init_checkpoint was loaded -- "
                  "training only the projector against a RANDOM, never-updated decoder "
                  "will not learn language. Freezing only makes sense once the decoder "
                  "already has a meaningful prior (text-pretrained or multimodal).")
        else:
            print("[model] decoder frozen -- training the projector only, against the "
                  "loaded checkpoint's language prior")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    proj_params = sum(p.numel() for p in model.projector.parameters())
    print(f"[model] parameters -- frozen: {frozen:,} | decoder: {dec_params:,} | "
          f"projector: {proj_params:,} | trainable total: {trainable:,}")
    return model


# ----------------------------------------------------------------------------
# Tokenizer setup: default is character-level (zero external downloads).
# Optionally, decoder.tokenizer can point at a real SentencePiece model file
# (llama2.c's `tokenizer.model` -- NOT its C-runtime-only `tokenizer.bin`,
# which SentencePieceProcessor cannot open) to use the same 32k-token
# vocabulary as stories15M, which is required for the init_checkpoint path.
# ----------------------------------------------------------------------------
def build_tokenizer(cfg: dict, train_ds_for_corpus):
    tok_path = cfg["decoder"].get("tokenizer")
    if tok_path and Path(tok_path).exists():
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=tok_path)

            class SPWrapper:
                bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), 0
                vocab_size = sp.vocab_size()

                def encode(self, text, max_len):
                    ids = [sp.bos_id()] + sp.encode(text)
                    ids = ids[: max_len - 1] + [sp.eos_id()]
                    return ids

                def decode(self, ids):
                    return sp.decode([i for i in ids if i not in (sp.bos_id(), sp.eos_id())])

            print(f"[tokenizer] loaded SentencePiece tokenizer: {tok_path} "
                  f"(vocab_size={sp.vocab_size()})")
            return SPWrapper()
        except Exception as e:
            print(f"[tokenizer] WARNING: could not load {tok_path} ({e}); falling back "
                  f"to the character-level tokenizer")

    corpus_chars = set()
    for i in range(min(2000, len(train_ds_for_corpus))):
        for cap in train_ds_for_corpus.all_captions(i):
            corpus_chars.update(cap)
    tok = CharTokenizer("".join(sorted(corpus_chars)))
    print(f"[tokenizer] using character-level tokenizer "
          f"(vocab_size={tok.vocab_size}) -- the default; no external download needed")
    return tok


# ----------------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------------
def get_lr(step: int, warmup_iters: int, base_lr: float, max_iters: int) -> float:
    if step < warmup_iters:
        return base_lr * (step + 1) / max(1, warmup_iters)
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def get_clip_features(model, pixel_values, row_indices, feat_cache, device):
    """Looks up cached CLIP features for this batch's image indices; runs
    CLIP only on the cache misses (new images this run hasn't embedded
    yet), then updates the cache. CLIP is frozen, so its output for a given
    image is identical every time it's asked for -- this just avoids paying
    for that identical forward pass more than once per image."""
    if feat_cache is None:
        return model.encode_image(pixel_values)
    hits, miss_pos = feat_cache.get(row_indices)
    if miss_pos:
        miss_pixels = pixel_values[miss_pos]
        computed = model.encode_image(miss_pixels)
        feat_cache.put(row_indices[miss_pos], computed)
    else:
        computed = torch.empty(0, device=device)
    return feat_cache.assemble(hits, miss_pos, computed, device)


@torch.no_grad()
def evaluate(model, loader, device, pad_id, feat_cache, max_batches=20):
    model.eval()
    losses = []
    for i, (pixel_values, ids, row_idx) in enumerate(loader):
        if i >= max_batches:
            break
        pixel_values, ids = pixel_values.to(device), ids.to(device)
        clip_feat = get_clip_features(model, pixel_values, row_idx, feat_cache, device)
        logits = model(pixel_values, ids, clip_features=clip_feat)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=pad_id
        )
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config, args)
    set_seed(cfg["data"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    print(f"[setup] device = {device} | seed = {cfg['data']['seed']}")

    print(f"[data] loading official splits of {cfg['data']['dataset']} "
          f"(cached under {cfg['data']['cache_dir']})")
    train_hf, cap_cols = load_flickr8k_split(cfg["data"]["dataset"], cfg["data"]["cache_dir"], "train")
    val_hf, _ = load_flickr8k_split(cfg["data"]["dataset"], cfg["data"]["cache_dir"], "validation")
    test_hf, _ = load_flickr8k_split(cfg["data"]["dataset"], cfg["data"]["cache_dir"], "test")
    print(f"[data] official image-level split: {len(train_hf)} train / {len(val_hf)} validation / "
          f"{len(test_hf)} test images ({len(cap_cols)} captions each). Splitting by IMAGE, before "
          f"picking a caption, keeps test images fully unseen -- their other captions never appear "
          f"in training either.")

    subset_size = cfg["data"].get("train_subset_size")
    if subset_size and subset_size < len(train_hf):
        official_train_size = len(train_hf)
        train_hf = train_hf.shuffle(seed=cfg["data"]["seed"]).select(range(subset_size))
        print(f"[data] using a fixed {subset_size}-image TRAINING subset (not the full "
              f"~{official_train_size}-image official split) so the lecture run does multiple "
              f"passes and the CLIP feature cache actually pays off within max_iters -- "
              f"see config comments / README 'Why 350 steps'.")

    from transformers import CLIPImageProcessor
    clip_processor = CLIPImageProcessor.from_pretrained(cfg["vision_encoder"]["name"])

    train_ds = CaptionDataset(train_hf, cap_cols, clip_processor, None,
                               cfg["data"]["max_caption_len"], mode="train", seed=cfg["data"]["seed"])
    tokenizer = build_tokenizer(cfg, train_ds)
    train_ds.tokenizer = tokenizer
    val_ds = CaptionDataset(val_hf, cap_cols, clip_processor, tokenizer,
                             cfg["data"]["max_caption_len"], mode="eval")

    if args.smoke_test:
        from torch.utils.data import Subset
        # Force a single, fixed caption per image ("eval" mode picks
        # caption_cols[0] deterministically) rather than the default
        # random-caption-per-epoch behavior. With 5 possible targets per
        # image, "loss should approach 0" is a fair sanity check only
        # against ONE fixed target -- otherwise irreducible caption-choice
        # entropy would keep loss from ever reaching ~0, even with a
        # perfectly correct model.
        train_ds.mode = "eval"
        train_ds = Subset(train_ds, list(range(8)))
        val_ds = Subset(val_ds, list(range(min(8, len(val_ds)))))
        print("[smoke-test] using 8 training examples, each with ONE fixed caption -- "
              "expect loss to fall close to 0 within a few hundred steps if the model "
              "and loss alignment are correct.")

    pad_id = getattr(tokenizer, "pad_id", 0)
    collate_fn = make_collate(pad_id)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )

    if cfg["data"].get("export_holdout", True) and not args.smoke_test:
        manifest_path, manifest = export_demo_holdout(
            test_hf, cap_cols, cfg["data"]["holdout_dir"],
            n=cfg["data"].get("holdout_n", 5), seed=cfg["data"]["seed"],
        )
        print(f"[data] exported {len(manifest)} holdout demo images + references to "
              f"{cfg['data']['holdout_dir']} (manifest: {manifest_path})")

    model = build_model(cfg, tokenizer.vocab_size, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                               weight_decay=cfg["train"]["weight_decay"])
    # Separate caches per split: row index 17 in train and row index 17 in
    # validation are two different images, so one shared cache keyed only
    # by row_idx would silently hand validation a training image's cached
    # CLIP features (or vice versa) whenever the indices happened to
    # coincide -- exactly the kind of bug that corrupts val_loss and
    # best.pt without ever raising an error.
    use_cache = cfg["train"].get("cache_clip_features", True)
    train_feat_cache = ClipFeatureCache() if use_cache else None
    val_feat_cache = ClipFeatureCache() if use_cache else None

    max_iters = cfg["train"]["max_iters"] or cfg["train"]["epochs"] * len(train_loader)
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.projector.load_state_dict(ckpt["projector"])
        if "decoder" in ckpt:
            model.decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        if "rng_snapshot" in ckpt:
            restore_rng_snapshot(ckpt["rng_snapshot"], train_ds)
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[resume] loaded {args.resume} at step {start_step} (optimizer state restored; "
              f"RNG restored for python/numpy/torch-cpu/torch-cuda/caption-sampling -- "
              f"see README for what this does and doesn't guarantee; "
              f"best_val_loss={best_val_loss:.4f})")

    print(f"[train] starting -- {max_iters} total optimizer steps "
          f"({cfg['train']['epochs']} epochs x {len(train_loader)} batches/epoch, "
          f"or --max-iters override)")

    step = start_step
    t0 = time.time()
    examples_seen = 0
    data_iter = iter(train_loader)
    while step < max_iters:
        try:
            pixel_values, ids, row_idx = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            pixel_values, ids, row_idx = next(data_iter)

        pixel_values, ids = pixel_values.to(device), ids.to(device)
        lr = get_lr(step, cfg["train"]["warmup_iters"], cfg["train"]["lr"], max_iters)
        for g in optim.param_groups:
            g["lr"] = lr

        clip_feat = get_clip_features(model, pixel_values, row_idx, train_feat_cache, device)
        logits = model(pixel_values, ids, clip_features=clip_feat)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), ignore_index=pad_id
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["train"]["grad_clip"])
        optim.step()

        examples_seen += pixel_values.shape[0]
        if step % cfg["train"]["log_every"] == 0:
            dt = time.time() - t0
            eps = examples_seen / max(dt, 1e-6)
            eta_s = (max_iters - step) * (dt / max(step, 1)) if step > 0 else float("nan")
            print(f"step {step:5d}/{max_iters} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{eps:.1f} examples/s | {dt:.1f}s elapsed | ETA {eta_s:.0f}s")

        completed_steps = step + 1  # `step` is this iteration's 0-indexed label;
                                     # by this point in the loop body, `completed_steps`
                                     # optimizer updates have actually happened -- this
                                     # is the number saved into checkpoints, so resuming
                                     # from e.g. "step100.pt" correctly means "100 updates
                                     # already done," not "about to redo update 100."
        if step > 0 and step % cfg["train"]["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device, pad_id, val_feat_cache)
            print(f"          [eval] val_loss {val_loss:.4f} "
                  f"(val feature cache: {len(val_feat_cache) if val_feat_cache else 0} images cached)")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optim, cfg, tokenizer, train_ds, completed_steps,
                                 best_val_loss, out_dir / "best.pt")
                print(f"          [ckpt] new best -- saved {out_dir / 'best.pt'}")

        if step > 0 and step % cfg["train"]["ckpt_every"] == 0:
            save_checkpoint(model, optim, cfg, tokenizer, train_ds, completed_steps,
                             best_val_loss, out_dir / f"step{completed_steps}.pt")
            save_checkpoint(model, optim, cfg, tokenizer, train_ds, completed_steps,
                             best_val_loss, out_dir / "last.pt")
            print(f"          [ckpt] saved {out_dir / f'step{completed_steps}.pt'} "
                  f"(train feature cache: {len(train_feat_cache) if train_feat_cache else 0} images cached)")

        step += 1

    save_checkpoint(model, optim, cfg, tokenizer, train_ds, step, best_val_loss, out_dir / "last.pt")
    print(f"[train] done. final checkpoint: {out_dir / 'last.pt'} "
          f"(best val_loss checkpoint, if any eval improved: {out_dir / 'best.pt'})")
    print("[train] run inferVLM.py against it, e.g.:")
    print(f"        python inferVLM.py --config {args.config} "
          f"--checkpoint {out_dir / 'best.pt'} --holdout-index 0")


def save_checkpoint(model: TinyVLM, optim: torch.optim.Optimizer, cfg: dict, tokenizer,
                     train_ds, step: int, best_val_loss: float, path: Path):
    # Decoder weights are saved UNCONDITIONALLY, even when train.freeze_decoder
    # is true: a frozen decoder still has *some* weights (random init, or an
    # opt-in llama2.c seed) that the projector was specifically trained
    # against, and inference needs to reconstruct that exact decoder rather
    # than silently substitute a freshly random-initialized one.
    proj_net = model.projector.net
    payload = {
        "projector": model.projector.state_dict(),
        "projector_config": {
            "clip_dim": proj_net[0].in_features,
            "hidden_dim": proj_net[0].out_features,
        },
        "decoder": model.decoder.state_dict(),
        "optimizer": optim.state_dict(),
        "rng_snapshot": get_rng_snapshot(train_ds),
        "step": step,
        "best_val_loss": best_val_loss,
        "decoder_config": cfg["decoder"],
        "vision_encoder": cfg["vision_encoder"]["name"],
    }
    if isinstance(tokenizer, CharTokenizer):
        payload["char_tokenizer_itos"] = tokenizer.itos
    else:
        payload["sentencepiece_model"] = cfg["decoder"].get("tokenizer")
    torch.save(payload, path)


if __name__ == "__main__":
    main()
