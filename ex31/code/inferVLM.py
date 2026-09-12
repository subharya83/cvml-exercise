#!/usr/bin/env python3
"""
inferVLM.py -- run the trained tiny VLM on one image and print a caption.

Usage:
    # against the exported holdout demo set (see trainVLM.py's export_demo_holdout)
    python inferVLM.py --config config.demo.yaml --checkpoint checkpoints/best.pt --holdout-index 0

    # against an arbitrary image
    python inferVLM.py --config config.demo.yaml --checkpoint checkpoints/best.pt --image photo.jpg

    # sampling instead of greedy decoding (greedy is the reproducible default)
    python inferVLM.py --config config.demo.yaml --checkpoint checkpoints/best.pt \
        --holdout-index 0 --sample --temperature 0.7 --top-k 20 --gen-seed 0

Meant for the closing "does it actually work" moment of the talk: point it at
a holdout image (one the model never saw during training) and read the
generated caption next to the five human references.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from tinyvlm_model import (CharTokenizer, DecoderConfig, TinyLlama, TinyVLM,
                            VisualProjector)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="config.demo.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="path to a .pt saved by trainVLM.py")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="path to an arbitrary image file")
    src.add_argument("--holdout-index", type=int,
                      help="index into the exported holdout manifest (see data.holdout_dir "
                           "in the config); also prints the 5 human reference captions")
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--sample", action="store_true",
                   help="use temperature/top-k sampling instead of the reproducible greedy default")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--gen-seed", type=int, default=None, help="RNG seed for --sample (ignored for greedy)")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_image_source(args, cfg):
    """Returns (image_path, reference_captions_or_None)."""
    if args.image:
        return args.image, None
    holdout_dir = Path(cfg["data"]["holdout_dir"])
    manifest_path = holdout_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run trainVLM.py first (it exports the holdout "
            f"demo set by default), or pass --image instead of --holdout-index."
        )
    manifest = json.loads(manifest_path.read_text())
    matches = [m for m in manifest if m["holdout_index"] == args.holdout_index]
    if not matches:
        raise ValueError(f"no holdout entry with index {args.holdout_index} in {manifest_path} "
                          f"(available: {[m['holdout_index'] for m in manifest]})")
    entry = matches[0]
    return str(holdout_dir / entry["image"]), entry["references"]


def main():
    args = build_argparser().parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(args.device or cfg["train"]["device"])
    max_new_tokens = args.max_new_tokens or cfg["infer"]["max_new_tokens"]
    temperature = args.temperature if args.temperature is not None else cfg["infer"]["temperature"]
    top_k = args.top_k if args.top_k is not None else cfg["infer"]["top_k"]

    image_path, references = resolve_image_source(args, cfg)

    print(f"[setup] device = {device}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    # weights_only=False: PyTorch >= 2.6 defaults torch.load to a restricted
    # unpickler that only allows plain tensors, rejecting the NumPy RNG-state
    # arrays/tuples this checkpoint's rng_snapshot carries (see
    # get_rng_snapshot() in trainVLM.py). Safe here because this checkpoint
    # was produced by THIS package's own trainVLM.py, not downloaded from a
    # third party -- never pass weights_only=False for a .pt file you didn't
    # create yourself or otherwise don't trust.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "decoder" not in ckpt and "char_tokenizer_itos" not in ckpt:
        # Ambiguous state -- fail loudly rather than silently falling back to
        # an uninitialized/mismatched decoder.
        raise RuntimeError(
            f"{args.checkpoint} has no saved decoder weights and no char-tokenizer vocabulary. "
            f"This checkpoint was likely trained with train.freeze_decoder=true and a "
            f"SentencePiece tokenizer -- point --config at the exact same tokenizer.model path "
            f"used during training, or retrain with the default (unfrozen decoder)."
        )
    print(f"[checkpoint] step={ckpt.get('step', '?')} best_val_loss={ckpt.get('best_val_loss', 'n/a')}")
    dcfg_dict = ckpt.get("decoder_config", cfg["decoder"])

    # Rebuild the tokenizer exactly as it was at training time.
    if "char_tokenizer_itos" in ckpt:
        tokenizer = CharTokenizer.__new__(CharTokenizer)
        tokenizer.itos = ckpt["char_tokenizer_itos"]
        tokenizer.stoi = {c: i for i, c in enumerate(tokenizer.itos)}
        tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id = 0, 1, 2
        vocab_size = len(tokenizer.itos)
        print(f"[tokenizer] restored character-level tokenizer (vocab_size={vocab_size})")
    else:
        import sentencepiece as spm
        sp_path = ckpt.get("sentencepiece_model") or cfg["decoder"]["tokenizer"]
        sp = spm.SentencePieceProcessor(model_file=sp_path)

        class SPWrapper:
            bos_id, eos_id, pad_id = sp.bos_id(), sp.eos_id(), 0
            vocab_size = sp.vocab_size()

            def decode(self, ids):
                return sp.decode([i for i in ids if i not in (sp.bos_id(), sp.eos_id())])

        tokenizer = SPWrapper()
        vocab_size = tokenizer.vocab_size
        print(f"[tokenizer] loaded SentencePiece tokenizer from {sp_path} (vocab_size={vocab_size})")

    from transformers import CLIPImageProcessor, CLIPModel

    vision_name = ckpt.get("vision_encoder", cfg["vision_encoder"]["name"])
    print(f"[model] loading frozen vision encoder: {vision_name}")
    clip_model = CLIPModel.from_pretrained(vision_name).to(device)
    clip_processor = CLIPImageProcessor.from_pretrained(vision_name)

    dcfg = DecoderConfig(
        dim=dcfg_dict["dim"], n_layers=dcfg_dict["n_layers"], n_heads=dcfg_dict["n_heads"],
        n_kv_heads=dcfg_dict["n_kv_heads"], vocab_size=vocab_size,
        max_seq_len=dcfg_dict["max_seq_len"], multiple_of=dcfg_dict["multiple_of"],
        dropout=0.0,
    )
    decoder = TinyLlama(dcfg)
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"])
        print("[model] loaded trained decoder weights from checkpoint")
    else:
        print("[model] WARNING: no decoder weights in checkpoint -- decoder was frozen during "
              "training and is running at its initial (random, or experimental llama2.c-seeded) "
              "weights, unchanged")

    # Prefer the checkpoint's own projector_config over the current YAML --
    # a checkpoint should load correctly even if config.demo.yaml's
    # projector.hidden_dim gets edited after training.
    proj_cfg = ckpt.get("projector_config", {
        "clip_dim": clip_model.config.projection_dim,
        "hidden_dim": cfg["projector"]["hidden_dim"],
    })
    projector = VisualProjector(
        clip_dim=proj_cfg["clip_dim"], decoder_dim=dcfg.dim,
        hidden_dim=proj_cfg["hidden_dim"],
    )
    projector.load_state_dict(ckpt["projector"])
    print(f"[model] loaded projector weights (trained for {ckpt.get('step', '?')} steps)")

    model = TinyVLM(decoder=decoder, projector=projector, clip_model=clip_model).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    generator = None
    if args.sample and args.gen_seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.gen_seed)

    generated_ids = model.generate(
        pixel_values, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        greedy=not args.sample, generator=generator,
    )
    caption = tokenizer.decode(generated_ids)

    print("\n" + "=" * 60)
    print(f"image      : {image_path}")
    print(f"decoding   : {'sampling (T=%.2f, top_k=%d)' % (temperature, top_k) if args.sample else 'greedy (reproducible)'}")
    print(f"caption    : {caption}")
    if references:
        print("references :")
        for r in references:
            print(f"  - {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()