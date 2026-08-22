# The Mathematics Behind Visual Language Models

Alumni talk for Asansol Engineering College undergraduate CS students, tracing how linear algebra, calculus, and optimization become a Visual Language Model — from a 2003 B.Tech CSE syllabus to systems like GPT-4V.

## Files

**`2026-08-21_AEC-LA-VLM-build_revised.pptx`**
20-slide deck. Opens with a live show-of-hands hook and a personal 2003-to-2026 framing, then moves through vectors, matrices, PCA, attention, vision encoding, the optimization loop, training stages, and efficiency tricks (LoRA), closing with a live interactive check and a callback to the opening story. Slide 7 (attention) presents cross-attention correctly: **Q** comes from the language tokens, **K** and **V** come from the image patch tokens — a word asks a question of the image, not of itself — and frames the softmax output as a probability distribution over patches.

**`2026-08-21_AEC-LA-VLM-build_revised.ipynb`**
Companion notebook for the live portion of the talk (Slide 19). No GPU needed; runs on CPU in a few seconds.

- **Part 1** — cosine similarity between a toy image embedding and matching/mismatched caption embeddings (Slide 4).
- **Part 2** — a concrete cross-attention walkthrough: 3 word tokens (`"a", "red", "car"`) and 2 image-patch tokens, with **Q** built only from the words and **K, V** built only from the patches, matching the corrected Slide 7 framing. Confirms attention weights form a proper probability distribution (rows sum to 1) and shows "red" and "car" attending most to `patch_1`.
- **Part 3** — heatmap of the word-by-patch attention matrix from Part 2.
- **Bonus (optional, needs internet)** — a real CLIP model computing actual image-caption similarity.

Run top to bottom; each cell builds on the last.

## Notes

- Both files should be presented/run together — the notebook is the live-coding companion to slides 4–8 and 19.
- If you re-run Part 2 with a different `np.random.seed(...)`, "a" may flip between patches (no strong signal built into its embedding) while "red" and "car" stay locked onto `patch_1`.
