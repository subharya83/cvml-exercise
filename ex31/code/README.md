# Build a Tiny VLM From Scratch

Companion code for the second Asansol Engineering College alumni lecture:
training and running a small Visual Language Model, live, on a laptop.

Nothing here is meant to produce a good VLM. It is meant to be small enough
to read end to end in an hour, and honest enough that the loss curve moving
on screen means something. This revision fixes a real correctness bug found
during review (see "Corrections from review" below) -- if you have an
earlier copy of this package, replace it rather than patching around it.

## What it is

```
image --[frozen CLIP encoder]--> pooled embedding --[trainable projector]-->
    1 "visual token" --[tiny Llama-style decoder]--> caption, token by token
```

- **Vision encoder**: `openai/clip-vit-base-patch32`, frozen. Public,
  pretrained, pre-downloadable. ~88M parameters. This is the model's "eyes"
  -- we never touch its weights, and it's kept in `.eval()` mode even
  across `model.train()` calls (see the model code).
- **Decoder**: a from-scratch, ~6.0M-parameter (with the default
  character-level tokenizer) reimplementation of Andrej Karpathy's
  llama2.c architecture (RMSNorm, rotary embeddings, SwiGLU, no biases,
  tied input/output embeddings). Trains from random initialization, end to
  end, jointly with the projector, on (image, caption) pairs. The
  "inspiration" from llama2.c that actually applies is the idea of a tiny
  transformer, trained from scratch, in a small hackable codebase -- not a
  specific checkpoint (see "A note on stories15M" below).
- **Projector**: the only piece with no pretraining anywhere in its
  history. A 2-layer MLP (512 -> 512 -> 288) mapping one CLIP embedding
  into the decoder's token-embedding space as a prefix token, exactly
  410,400 parameters.
- **Data**: [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k),
  using its **official, image-level** train / validation / test splits
  (~6,000 / ~1,000 / ~1,000 images, 5 caption columns each) -- not a
  re-split of the flattened (image, caption) pairs. That distinction
  matters: splitting after flattening lets near-duplicate captions of the
  same photo land on both sides of the split, so a "held-out" image would
  almost always have already been seen with a different caption attached.

## Corrections from review

An earlier version of this package had two real bugs and several
overclaimed slide numbers, caught in review before the lecture was given.
Recorded here rather than quietly fixed, because the mistakes are
instructive:

1. **Teacher-forcing was off by one position.** `TinyVLM.forward()` fed the
   decoder `[visual, BOS, tok_1, ..., tok_{T-2}]` and compared its output
   against `caption_ids[:, 1:]` using `logits[:, :-1, :]` -- but decoder
   position *i* predicts sequence position *i+1*, so the correct slice is
   `logits[:, 1:, :]`. The old code trained every position against its
   *predecessor's* label instead of its own next token. Training loss still
   decreased (the model could partly cheat on position/frequency
   statistics), which is exactly why this kind of bug is dangerous: nothing
   looked wrong until someone checked whether generated captions made
   sense. Fixed in `tinyvlm_model.py`; a `--smoke-test` mode was added to
   `trainVLM.py` specifically to catch this class of bug in 30 seconds
   (overfit 8 examples; loss should approach ~0).
2. **The split was caption-level, not image-level.** The five captions of
   one photo were expanded into five separate (image, caption) examples
   *before* the 80/10/10 split, so siblings of a "held-out" image routinely
   ended up in training. Fixed by using Flickr8k's own official image-level
   splits directly (see "What it is" above).
3. Several slide numbers (step counts, parameter counts) were estimates
   that didn't match what the code actually does. They're now computed
   from the instantiated model / config rather than guessed -- see
   "Parameter counts" and "Why 350 steps" below.

A second review pass caught two more, after the first three were fixed:

4. **The CLIP feature cache was shared between the training and validation
   splits, keyed only by row index.** Row 17 of the train split and row 17
   of the validation split are different images, but they wrote to and
   read from the same cache slot -- so validation could silently score
   against a training image's cached CLIP embedding (or vice versa),
   corrupting `val_loss` and therefore which checkpoint got saved as
   `best.pt`. Fixed with two separate `ClipFeatureCache` instances, one per
   split. This hit the `--smoke-test` mode particularly badly, since its 8
   validation rows would have reused its 8 training rows' cached features
   outright.
5. **The feature cache didn't actually help the 350-step demo run.** Caching
   only pays off when the *same* image is looked up more than once. Over
   the full ~6,000-image official train split, 350 steps at batch 16 sees
   roughly 5,600 largely distinct images -- almost every lookup a cache
   miss -- so the earlier claim that caching was "most of what makes 350
   steps fit" was not true for training. Fixed by training on a fixed,
   seeded 1,500-image subset (`data.train_subset_size`) instead, so 350
   steps is ~3.7 passes over the same images and the cache is genuinely
   warm for most of the run. See "Why 350 steps" below.

## Files

| File                | Purpose                                                       |
|---------------------|----------------------------------------------------------------|
| `hardware.sh`       | Environment check: hardware, packages, caches, a CLIP micro-benchmark |
| `config.demo.yaml`  | The exact profile used in the lecture (350 steps)              |
| `config.full.yaml`  | A longer, non-lecture training run                             |
| `config.yaml`       | Superseded -- do not use (kept only because it can't be deleted) |
| `tinyvlm_model.py`  | The architecture: TinyLlama decoder, projector, llama2.c loader |
| `tinyvlm_data.py`   | Official-split loading, lazy Dataset, CLIP feature cache, holdout export |
| `trainVLM.py`       | Training script (run this first, in a terminal)                |
| `inferVLM.py`       | Loads a checkpoint, captions a holdout or arbitrary image       |
| `visualize.py`      | Plots `checkpoints/metrics.csv` -- live, in a second terminal, or after the fact |

## 1. Check your hardware and environment

```bash
chmod +x hardware.sh
./hardware.sh
```

Works on both macOS (Intel and Apple Silicon) and Linux. Reports CPU, RAM,
GPU/MPS availability, free disk, whether the required Python packages are
installed, whether Flickr8k and the CLIP weights are already cached, output
directory writability, and a short CLIP forward-pass timing benchmark once
those packages/weights are present.

## 2. Install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.0 torchvision==0.18.0 transformers==4.41.0 \
    datasets==2.19.0 sentencepiece==0.2.0 pyyaml==6.0.1 pillow==10.3.0 \
    matplotlib==3.8.4
```
(`matplotlib` is only needed for `visualize.py` -- `trainVLM.py` and
`inferVLM.py` don't import it.)
(Pin these, or nearby versions, rather than "latest" -- `datasets` and
`transformers` both move fast enough to occasionally change default
behavior between minor versions.)

## 3. Pre-fetch the dataset and CLIP weights (one-time, before the talk)

No decoder checkpoint download is needed for the default path -- the
decoder trains from random initialization. Two things are still worth
prefetching so the talk doesn't stall on wifi:

```bash
python3 -c "
from datasets import load_dataset
for split in ('train', 'validation', 'test'):
    load_dataset('jxie/flickr8k', cache_dir='./data/flickr8k', split=split)
"
python3 -c "
from transformers import CLIPModel, CLIPImageProcessor
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
"
```

The Flickr8k dataset is cached under `./data/flickr8k` (~1.1 GB); the CLIP
checkpoint is cached in the standard Hugging Face cache
(`~/.cache/huggingface`, ~600 MB) -- prefetching only the former and
assuming the pipeline is "offline-ready" was an earlier oversight in this
package. `hardware.sh` checks both caches.

## 4. Five-minute smoke test

Before a real run (and especially before trusting any change you make to
`tinyvlm_model.py`), run:

```bash
python trainVLM.py --config config.demo.yaml --smoke-test
```

This overfits 8 training examples for a few hundred steps. Expected output:
loss starting somewhere around `ln(vocab_size)` (≈4.3 for a ~75-character
vocabulary) and falling toward ~0 within a few hundred steps. If it
plateaus well above 0, something in the model or the loss alignment is
broken -- that's exactly the check that would have caught the
teacher-forcing bug above in 30 seconds instead of after a full lecture
run.

## 5. Train

```bash
python trainVLM.py --config config.demo.yaml
```

This is **the exact command and config used throughout the lecture** --
350 optimizer steps, batch size 16, over Flickr8k's ~6,000-image official
train split with one caption sampled per image per epoch (so 6000 / 16 =
375 iterations per epoch; 350 steps is just under one full pass). See "Why
350 steps" below for the arithmetic behind that number.

Useful overrides (see `python trainVLM.py --help` for all of them):

```bash
# a shorter demo if you're tight on time
python trainVLM.py --config config.demo.yaml --max-iters 150

# resume a checkpoint -- restores weights, optimizer state, step count, and
# best-effort RNG state (see "Resume: what's actually reproducible" below)
python trainVLM.py --config config.demo.yaml --resume checkpoints/last.pt

# freeze the decoder and train only the projector -- only sensible once the
# decoder already has a meaningful language prior (text-pretrained via
# init_checkpoint, or multimodal); with the default random-init decoder,
# freezing it prevents the model from learning language at all
python trainVLM.py --config config.demo.yaml --freeze-decoder
```

The script logs `step | loss | lr | examples/s | elapsed | ETA` every
`train.log_every` steps, runs validation every `train.eval_every` steps
(saving `checkpoints/best.pt` whenever validation loss improves), and
checkpoints to `train.out_dir` every `train.ckpt_every` steps plus a
rolling `last.pt`. It also prints the actual instantiated parameter counts
(frozen / decoder / projector / trainable total) at startup -- see
"Parameter counts" below for what those numbers should be.

On first run, it also exports a small holdout demo set from the **test**
split (never used for training or checkpoint selection) to `./demo/`:
`holdout_0000.jpg` ... `holdout_0004.jpg`, matching `_references.txt`
files, and a `manifest.json` tying them together. `inferVLM.py` reads
this directly.

## Watch the loss curve live

`trainVLM.py` appends one row per logged training step and one row per
validation pass to `checkpoints/metrics.csv` (path configurable via
`train.metrics_csv`), flushing immediately. While training is running in
one terminal, plot it live from a second one:

```bash
python visualize.py --config config.demo.yaml --watch
```

This is exactly what slide 10 and 11's terminal output is standing in for
-- a real, continuously-updating matplotlib window with raw train loss,
a smoothed train-loss line, validation loss at each eval point, and a
dashed line marking the best validation loss seen so far. Ctrl+C to stop
watching (training keeps running independently). After training finishes,
drop `--watch` for a single static plot, or add `--out loss_curve.png` to
save a file instead of (or in addition to) showing a window:

```bash
python visualize.py --config config.demo.yaml --out loss_curve.png
```

`--smooth N` controls the rolling-average window over the noisy raw train
loss (default 5; the raw per-batch line is also plotted underneath, faded,
so the smoothing is never hiding anything). `--csv path/to/metrics.csv`
points at a specific file directly if you'd rather not go through a config
(useful for comparing two runs' CSVs side by side by copying them to
different filenames first).

## 6. Run inference

```bash
# against the exported holdout demo set
python inferVLM.py --config config.demo.yaml --checkpoint checkpoints/best.pt --holdout-index 0

# against an arbitrary image
python inferVLM.py --config config.demo.yaml --checkpoint checkpoints/best.pt --image path/to/photo.jpg
```

Greedy decoding is the **default** -- deterministic, so the same command
reproduces the same caption every time, which matters live in front of a
class. Pass `--sample --temperature 0.7 --top-k 20 --gen-seed 0` to
demonstrate sampling as a separate, explicitly optional comparison.
`--holdout-index` also prints the five human reference captions next to
the generated one.

## Resume: what's actually reproducible

`--resume` saves and restores optimizer state, step count, and a best-effort
RNG snapshot: Python's `random` state (which drives both `CaptionDataset`'s
per-epoch caption sampling and, since no dedicated generator is passed, the
default `DataLoader` shuffle order too), NumPy's state, and PyTorch's CPU and
CUDA generator states. That covers a single-process run well. It is **not**
a guarantee of bit-for-bit identical resume under multi-worker
(`num_workers > 0`) `DataLoader` reshuffling, which does its own per-worker
seeding that this package doesn't snapshot. Treat `--resume` as "continues
training sensibly from where it left off," not "replays an identical
sequence of batches." Checkpoints also store `step + 1` (completed
optimizer updates), not the raw loop counter, so resuming from
`step100.pt` correctly means "100 updates already done" rather than
re-running update 100.

Every checkpoint is also self-contained: it stores its own
`projector_config` (the exact `clip_dim`/`hidden_dim` it was built with)
and decoder weights (saved unconditionally, even when `freeze_decoder` was
true, since a frozen decoder still has specific weights -- random-init or
an experimental llama2.c seed -- that the projector was trained against).
`inferVLM.py` prefers these stored values over the current YAML, so editing
`config.demo.yaml` after training doesn't break loading an old checkpoint.

## Parameter counts

Computed directly from the instantiated classes in `tinyvlm_model.py`
(dim=288, n_layers=6, multiple_of=32, tied embeddings):

| Component                              | Parameters   |
|-----------------------------------------|-------------:|
| CLIP ViT-B/32 vision encoder (frozen)   | ~88,000,000  |
| Decoder, character tokenizer (default, vocab ≈ 75) | ~6,000,000 |
| Decoder, if a 32k-vocab SentencePiece tokenizer is used | ~15,200,000 |
| Projector (512 → 512 → 288, with biases) | 410,400     |

`decoder.vocab_size` does not appear as a fixed number in
`config.demo.yaml` / `config.full.yaml` on purpose: it is always replaced
at startup by the active tokenizer's real vocabulary size (see
`build_tokenizer()` in `trainVLM.py`), so hardcoding a value there would
describe a model that isn't the one actually built.

## Why 350 steps (and why a 1,500-image subset)

```
official train split (image-level)  : ~6,000 images
data.train_subset_size              : 1,500 images (fixed, seeded subset)
batch size                          : 16
iterations per epoch (over subset)  : 1500 / 16 ~ 94
config.demo.yaml max_iters          : 350   (~ 3.7 passes over the subset)
```

Two things had to both be true for this to be a good in-class run: it
needed to fit inside a lecture slot, and the CLIP feature cache (see
`train.cache_clip_features`) needed to actually do something. Running 350
steps over the *full* ~6,000-image split fails the second requirement:
at batch 16, 350 steps touches roughly 5,600 images, and since almost none
of them repeat within a single (incomplete) epoch, nearly every lookup
would be a cache miss -- CLIP would run in full for nearly every step, and
the "caching is what makes this fit the time budget" claim would be false
advertising. Training on a fixed 1,500-image subset instead means the
cache is fully warm by the end of the first pass (~step 94), and the
remaining ~256 steps are mostly cache hits -- and students get to watch
multiple real epochs finish inside one sitting, not a fraction of one.
`config.full.yaml` trains on the complete official split instead, for a
longer, non-lecture run.

## A note on stories15M (and why it's not the default)

The original idea for this build was to reuse Andrej Karpathy's llama2.c
`stories15M.bin` -- a real, small, fully-trained checkpoint -- as a head
start for the decoder. It doesn't transfer as directly as you'd hope:
`stories15M` was trained purely on TinyStories *text*, predicting the next
token from other text tokens only, and never saw a visual prefix token. But
"it has no visual grounding" does not mean "it has nothing to offer" -- it
still carries a real prior over English grammar, vocabulary, and common
next-token statistics that a random-init decoder has to learn from
scratch. The honest, defensible claim is narrower than either extreme:

> This *particular* checkpoint may transfer poorly for reasons specific to
> it -- small scale, a narrow text domain (children's stories), and a
> tokenizer/vocabulary that has to match exactly. Whether it measurably
> helps this captioning task is a question to test, not assume either way.

To make that test actually runnable, two things needed fixing:

- **The tokenizer file was wrong.** llama2.c ships two different files:
  `tokenizer.model` (a standard SentencePiece protobuf, the same one used
  to train `stories15M`) and `tokenizer.bin` (a separate, custom binary
  export meant only for llama2.c's C inference runtime).
  `sentencepiece.SentencePieceProcessor` can load the former, not the
  latter -- the earlier version of this package pointed at `tokenizer.bin`
  and would have silently fallen back to the character tokenizer every
  time, making the whole experiment unreachable. Point `decoder.tokenizer`
  at `tokenizer.model` instead:

  ```bash
  mkdir -p weights
  curl -L -o weights/stories15M.bin \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
  curl -L -o weights/tokenizer.model \
    https://github.com/karpathy/llama2.c/raw/master/tokenizer.model
  ```

  then in a copy of `config.demo.yaml`:

  ```yaml
  decoder:
    init_checkpoint: ./weights/stories15M.bin
    tokenizer: ./weights/tokenizer.model
  ```

- **It's framed as a comparison, not a default.** Run it side by side with
  a random-init run over the same number of steps and compare validation
  loss curves. That comparison is more interesting, and more honest, than
  asserting an answer on a slide.

## Extending this further

1. **Run the stories15M comparison** above and bring the two loss curves to
   class.
2. **Freeze the decoder** (`train.freeze_decoder: true`) once it has a
   meaningful language prior to freeze -- either the stories15M experiment
   above (a real, if visually-ungrounded, text prior) or a real pretrained
   *multimodal* checkpoint. This is in fact a standard alignment-tuning
   pattern: freeze a capable language model, train only an adapter to align
   a new modality into its embedding space. With the default random-init
   decoder, though, freezing it trains nothing useful -- there's no
   language prior yet for the projector to align into.
3. **More visual tokens**: project CLIP's patch-level features (not just
   the pooled embedding) into a longer visual prefix. This is still
   self-attention over a longer sequence, not cross-attention -- true
   cross-attention needs separate text queries and image keys/values (or
   dedicated cross-attention layers), which this build doesn't implement.
4. **Bigger / different data**: COCO Captions, LAION subsets, or your own
   photos -- but expect to write a small adapter for `tinyvlm_data.py`
   rather than a drop-in swap; the current loader assumes Flickr8k's
   specific column names and official splits.
5. **Beyond captioning**: visual question answering needs more than
   prepending a question to the caption target -- a real input/target
   convention, loss masking so the question tokens aren't penalized as
   generation targets, and VQA-shaped training data.
6. **Quantize for deployment**: llama2.c's `run.c` can run the *decoder*
   in pure C with no Python runtime, but that alone is not an end-to-end
   deployable VLM -- the CLIP vision tower and the projector would also
   need their own export/runtime path (e.g. ONNX or Core ML for CLIP).

## What this build takes as a shortcut (and what production VLMs do differently)

Being explicit about the gap, rather than letting it hide in vague words
like "tiny":

| Here                                          | A production VLM                                  |
|------------------------------------------------|-----------------------------------------------------|
| Character-level tokenizer by default            | A trained subword tokenizer (BPE/SentencePiece) over a large corpus |
| One pooled CLIP embedding as a single visual token | Many patch tokens, often with real cross-attention |
| ~6-15M decoder parameters                       | Billions                                            |
| ~6,000 training images                          | Hundreds of millions to billions of image-text pairs |
| Minutes of CPU/MPS training                     | Weeks on GPU/TPU clusters                           |
| No KV-cache during generation                   | KV-caching for practical inference latency          |
| No formal caption metric (BLEU/CIDEr/etc.)      | Standard benchmarks and human evaluation            |
| No attention mask                               | Explicit padding/attention masks for arbitrary batching |

That last row is deliberate, not an oversight: this build always pads on
the right and always attends causally, so a padded tail can never leak into
an earlier token's prediction, and `ignore_index=pad_id` already excludes
every padded position from the loss -- no mask changes the result. A
production system that pads on the left, batches non-causally, or needs the
model to know which positions were real at inference time would need one;
`tinyvlm_data.py` has a comment at the collate function marking exactly
where to add it back if you extend the code in that direction.

## Troubleshooting

- **`MPS backend out of memory` / MPS errors on Apple Silicon**: pass
  `--device cpu`; this build is CPU-tested and CPU-timed by design.
- **CUDA out of memory**: lower `train.batch_size` in the config, or pass
  `--batch-size 8`.
- **`DataLoader` hangs or errors with `num_workers > 0` on macOS**: this
  package already avoids `lambda` collate functions (they can't be pickled
  by macOS/Windows spawn-based multiprocessing) -- if you've customized
  the collate function yourself, keep it a top-level function, not a
  lambda or closure.
- **`FileNotFoundError` on `demo/manifest.json`**: run `trainVLM.py` at
  least once first (it exports the holdout set by default), or pass
  `--image` instead of `--holdout-index` to `inferVLM.py`.
- **`_pickle.UnpicklingError: Weights only load failed` / mentions
  `numpy._core.multiarray._reconstruct`**: you're on PyTorch >= 2.6, which
  changed `torch.load`'s default from `weights_only=False` to
  `weights_only=True` -- a restricted unpickler that only allows plain
  tensors. Our checkpoints also carry a `rng_snapshot` (NumPy RNG state
  arrays/tuples; see "Resume: what's actually reproducible" above), which
  that restricted mode rejects. Both `trainVLM.py` (`--resume`) and
  `inferVLM.py` already pass `weights_only=False` explicitly for exactly
  this reason -- if you still hit this, you're likely running an older
  copy of one of those two files; re-copy them from this package rather
  than patching the error away yourself. (`weights_only=False` is only
  safe because these checkpoints are ones this same code produced --
  never load a `.pt` file from an untrusted source that way.)
- **Checkpoint loads but generates garbage / empty captions**: run
  `python trainVLM.py --config config.demo.yaml --smoke-test` first. If
  loss doesn't fall toward ~0 on 8 overfit examples, the bug is in the
  model/loss, not in the amount of real training data or steps.
- **Corrupted or partial checkpoint file**: `torch.load` will raise
  clearly; re-run training from the last known-good `step*.pt`, or from
  scratch -- checkpoints are cheap here (a few hundred MB at most).

## Mapping back to the Part 1 (mathematics) lecture

| Part 1 concept       | Where it shows up here                                    |
|------------------------|-------------------------------------------------------------|
| `y = Wx + b`            | The two `nn.Linear` layers inside `VisualProjector`        |
| Embeddings              | The CLIP pooled vector, and `TinyLlama.tok_embeddings`      |
| Attention               | Causal self-attention over `[visual_token, caption_tokens]` in `Attention.forward()` -- a visual **prefix**, not cross-attention |
| Loss minimization       | `F.cross_entropy(logits, caption_ids[:, 1:])`               |
| Backpropagation         | `loss.backward()`                                            |
| Adam                    | `torch.optim.AdamW`                                          |
| Inference               | `TinyVLM.generate()` -- repeated next-token prediction        |

## Notes on scope

This is a pedagogical build, not a usable product. Expect the tiny decoder
(6 layers, 288 dim, character-level tokenizer) to produce captions that are
grammatically loose and only loosely tied to image content after a
350-step in-class run -- that gap between "it visibly worked" and "it
works well" is itself the lesson: scale, data quality, and training time
are most of what separates this from a real VLM, not architecture
cleverness. A simple way to make that gap visible live: compare generated
captions from (a) the trained model on a holdout image, (b) the same model
given a *shuffled* visual embedding (a different image's CLIP features),
and (c) an image-independent decoder run. If (a) differs meaningfully from
(b) and (c), that's evidence the model is actually using the image, not
just reciting a common Flickr-style sentence.