"""
tinyvlm_data.py
----------------
Loads Flickr8k using its OFFICIAL, image-level train/validation/test splits
(jxie/flickr8k on the Hugging Face Hub ships all three: ~6,000 / ~1,000 /
~1,000 images, 5 caption columns per image). Splitting at the image level,
before any caption is picked, matters: five captions of the same photo are
near-duplicates of each other, so splitting AFTER flattening to (image,
caption) pairs lets near-duplicates of a "held-out" photo leak into
training -- almost every nominally unseen image would in fact have been
seen, just with a different caption attached. Using the dataset's own
splits sidesteps that by construction.

Also provides:
  - a lazy Dataset wrapper (keeps the HF dataset's on-disk arrow backing;
    never decodes/holds thousands of PIL images in a Python list),
  - a ClipFeatureCache that reuses one image's CLIP embedding across the
    caption(s) and epoch(s) it appears in, since CLIP is frozen and its
    output for a given image never changes,
  - export_demo_holdout(), which writes a manifest.json plus one image file
    and its five reference captions to disk, so `inferVLM.py --holdout-index
    N` has something concrete to load instead of an in-memory split that
    disappears when the training process exits.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset


CAPTION_COL_CANDIDATES = ["caption_0", "caption_1", "caption_2", "caption_3", "caption_4"]


def _caption_columns(column_names: list[str]) -> list[str]:
    cols = [c for c in column_names if "caption" in c.lower()]
    if not cols:
        raise ValueError(f"no caption-like columns found in {column_names}")
    return sorted(cols)


def load_flickr8k_split(dataset_id: str, cache_dir: str, split: str):
    """Loads one of Flickr8k's official splits ("train", "validation", or
    "test") WITHOUT decoding any images -- returns the lazy HF `Dataset`
    object plus the list of caption column names."""
    from datasets import load_dataset  # local import: only needed here

    ds = load_dataset(dataset_id, cache_dir=cache_dir, split=split)
    caption_cols = _caption_columns(ds.column_names)
    return ds, caption_cols


class CaptionDataset(Dataset):
    """Wraps ONE official split lazily: each `__getitem__` decodes exactly
    one image (via the CLIP processor) and picks exactly one of its five
    captions, so nothing is expanded into a bigger in-memory list.

    - `mode="train"`: picks a *different random caption* each time an index
      is drawn (a cheap, free form of caption augmentation across epochs).
    - `mode="eval"` (validation/test/holdout): always picks the same
      caption (`caption_cols[0]`), so evaluation numbers are reproducible
      run to run.
    """

    def __init__(self, hf_dataset, caption_cols: list[str], clip_processor, tokenizer,
                 max_caption_len: int, mode: str = "train", seed: int = 1337):
        assert mode in ("train", "eval")
        self.ds = hf_dataset
        self.caption_cols = caption_cols
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        self.mode = mode
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ds)

    def all_captions(self, i: int) -> list[str]:
        row = self.ds[i]
        return [row[c].strip() for c in self.caption_cols if row.get(c)]

    def __getitem__(self, i):
        row = self.ds[i]
        captions = [row[c].strip() for c in self.caption_cols if row.get(c)]
        caption = self.rng.choice(captions) if self.mode == "train" else captions[0]

        pixel_values = self.clip_processor(
            images=row["image"].convert("RGB"), return_tensors="pt"
        )["pixel_values"][0]
        ids = self.tokenizer.encode(caption, self.max_caption_len)
        return pixel_values, torch.tensor(ids, dtype=torch.long), i


def collate(batch, pad_id: int):
    """Top-level (picklable) collate function -- a `lambda` here would
    silently break `num_workers > 0` under macOS/Windows spawn-based
    multiprocessing, since lambdas can't be pickled.

    No attention mask: this build deliberately doesn't need one. Padding is
    always on the RIGHT, attention is always CAUSAL (a position only ever
    looks at itself and earlier positions), so a padded tail can never
    leak into an earlier, real token's prediction -- and the loss already
    excludes every padded position via `ignore_index=pad_id`. An attention
    mask matters once you either pad on the left, batch variable-length
    sequences non-causally, or want the model to see which positions were
    real at inference time; none of that applies here. If you extend this
    code in a direction where any of those stop being true, add the mask
    back and wire it into `Attention.forward()`."""
    pixel_values = torch.stack([b[0] for b in batch])
    max_len = max(b[1].shape[0] for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, (_, cap, _idx) in enumerate(batch):
        ids[i, : cap.shape[0]] = cap
    row_indices = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return pixel_values, ids, row_indices


def make_collate(pad_id: int):
    return partial(collate, pad_id=pad_id)


# ----------------------------------------------------------------------------
# CLIP feature cache: since CLIP is frozen, its embedding for a given image
# never changes across captions or epochs -- computing it once per image and
# reusing it turns most of an epoch's forward passes into "projector + tiny
# decoder only", which is most of what makes this fit a lecture's time
# budget on a CPU/MPS laptop.
# ----------------------------------------------------------------------------
class ClipFeatureCache:
    def __init__(self):
        self._cache: dict[int, torch.Tensor] = {}

    def get(self, row_indices: torch.Tensor):
        """Returns (cached_features_or_None_per_row, indices_needing_compute).
        cached_features_or_None_per_row is a list aligned with row_indices."""
        hits, miss_pos = [], []
        for pos, idx in enumerate(row_indices.tolist()):
            v = self._cache.get(idx)
            hits.append(v)
            if v is None:
                miss_pos.append(pos)
        return hits, miss_pos

    def put(self, row_indices: torch.Tensor, features: torch.Tensor):
        for idx, feat in zip(row_indices.tolist(), features):
            self._cache[idx] = feat.detach().cpu()

    def assemble(self, hits, miss_pos, computed: torch.Tensor, device):
        """Fills in the freshly computed features at miss_pos and stacks
        everything back into batch order, on `device`."""
        out = list(hits)
        for pos, feat in zip(miss_pos, computed):
            out[pos] = feat.detach().cpu()
        return torch.stack(out).to(device)

    def __len__(self):
        return len(self._cache)


# ----------------------------------------------------------------------------
# Holdout demo export: a concrete, on-disk artifact for the live "run
# inference on something the model never trained on" moment, instead of an
# in-memory split that only exists inside the training process.
# ----------------------------------------------------------------------------
def export_demo_holdout(test_ds, caption_cols: list[str], out_dir: str, n: int = 5,
                         seed: int = 1337):
    """Saves `n` images from the TEST split (never used for training or
    checkpoint selection) plus their reference captions and a manifest, so
    `inferVLM.py --holdout-index K` can load exactly what the slides show."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idxs = list(range(len(test_ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n]

    manifest = []
    for rank, i in enumerate(idxs):
        row = test_ds[i]
        img_path = out / f"holdout_{rank:04d}.jpg"
        row["image"].convert("RGB").save(img_path)
        refs = [row[c].strip() for c in caption_cols if row.get(c)]
        ref_path = out / f"holdout_{rank:04d}_references.txt"
        ref_path.write_text("\n".join(refs) + "\n")
        manifest.append({
            "holdout_index": rank,
            "source_split_row": i,
            "image": img_path.name,
            "references_file": ref_path.name,
            "references": refs,
        })

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path, manifest
