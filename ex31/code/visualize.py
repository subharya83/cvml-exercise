#!/usr/bin/env python3
"""
visualize.py -- plot the training/validation loss curve trainVLM.py writes
to metrics.csv, live or after the fact.

trainVLM.py appends one row per log_every training step (split=train) and
one row per eval_every validation pass (split=val), flushing immediately --
so this script can either plot a finished run, or run alongside a live one
in a second terminal.

Usage:
    # after training finishes: one static plot, saved to a file
    python visualize.py --csv checkpoints/metrics.csv --out loss_curve.png

    # DURING training, in a second terminal: auto-refreshing live plot
    python visualize.py --csv checkpoints/metrics.csv --watch

    # from the config instead of typing the csv path out
    python visualize.py --config config.demo.yaml --watch
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import yaml


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=str, default=None,
                   help="path to metrics.csv (default: derived from --config)")
    p.add_argument("--config", type=str, default="config.demo.yaml",
                   help="used to find metrics.csv when --csv is omitted "
                        "(train.metrics_csv, or train.out_dir/metrics.csv)")
    p.add_argument("--out", type=str, default=None,
                   help="save the plot to this file instead of (or in addition to, "
                        "with --watch off) showing it interactively")
    p.add_argument("--watch", action="store_true",
                   help="poll the CSV and keep redrawing -- run this in a second "
                        "terminal while trainVLM.py is still running")
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between redraws in --watch mode (default: 2)")
    p.add_argument("--smooth", type=int, default=5,
                   help="rolling-average window over TRAIN loss points, to cut through "
                        "batch-to-batch noise (default: 5; 1 disables smoothing)")
    return p


def resolve_csv_path(args) -> Path:
    if args.csv:
        return Path(args.csv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    explicit = cfg["train"].get("metrics_csv")
    if explicit:
        return Path(explicit)
    return Path(cfg["train"]["out_dir"]) / "metrics.csv"


def read_metrics(csv_path: Path):
    """Returns (train_steps, train_losses, val_steps, val_losses, best_val_loss).
    Tolerant of a CSV that's mid-write (an in-progress row from --watch
    polling a live run): any short/partial trailing row is just skipped."""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    best_val_loss = None
    if not csv_path.exists():
        return train_steps, train_losses, val_steps, val_losses, best_val_loss
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["step"])
                split = row["split"]
                loss = float(row["loss"])
            except (KeyError, ValueError, TypeError):
                continue  # partially-flushed trailing row while a live run writes it
            if split == "train":
                train_steps.append(step)
                train_losses.append(loss)
            elif split == "val":
                val_steps.append(step)
                val_losses.append(loss)
                bv = row.get("best_val_loss")
                if bv not in (None, ""):
                    try:
                        best_val_loss = float(bv)
                    except ValueError:
                        pass
    return train_steps, train_losses, val_steps, val_losses, best_val_loss


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 2:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo: i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def draw(ax, csv_path: Path, smooth: int):
    ax.clear()
    train_steps, train_losses, val_steps, val_losses, best_val_loss = read_metrics(csv_path)

    if not train_steps and not val_steps:
        ax.set_title(f"Waiting for data in {csv_path} ...")
        return

    if train_steps:
        ax.plot(train_steps, train_losses, color="#B0B8C4", linewidth=1, alpha=0.6,
                 label="train loss (raw)")
        smoothed = rolling_mean(train_losses, smooth)
        ax.plot(train_steps, smoothed, color="#1C7293", linewidth=2,
                 label=f"train loss (smoothed, window={smooth})" if smooth > 1 else "train loss")

    if val_steps:
        ax.plot(val_steps, val_losses, color="#F2A541", linewidth=2, marker="o",
                 markersize=5, label="validation loss")

    if best_val_loss is not None:
        ax.axhline(best_val_loss, color="#4C9A2A", linewidth=1, linestyle="--",
                    label=f"best val_loss = {best_val_loss:.4f}")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("tinyVLM training curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)


def main():
    args = build_argparser().parse_args()
    csv_path = resolve_csv_path(args)

    import matplotlib
    if args.out and not args.watch:
        matplotlib.use("Agg")  # headless: no display needed just to save a file
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if args.watch:
        plt.ion()
        print(f"[visualize] watching {csv_path} -- Ctrl+C to stop")
        try:
            while True:
                draw(ax, csv_path, args.smooth)
                fig.canvas.draw()
                fig.canvas.flush_events()
                if args.out:
                    fig.savefig(args.out, dpi=150, bbox_inches="tight")
                plt.pause(args.interval)
        except KeyboardInterrupt:
            print("\n[visualize] stopped")
    else:
        draw(ax, csv_path, args.smooth)
        if args.out:
            fig.savefig(args.out, dpi=150, bbox_inches="tight")
            print(f"[visualize] saved {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()