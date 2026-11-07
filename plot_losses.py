"""Make training_loss.png and validation_loss.png from the FastSpeech 2 log files.

Usage:
  python plot_losses.py --train <path>/log/train/log.txt --val <path>/log/val/log.txt \
      --out ../Results/English --name English

Lines are expected to look like
  Step 100/30000, Total Loss: 1.23, Mel Loss: ..., Pitch Loss: ..., ...
  Validation Step 500, Total Loss: 1.10, Mel Loss: ...
The parser only needs the step number and the "<Name> Loss: <number>" pairs.
Also prints a short summary (final, best, gap) for the report.
"""
import os
import re
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"[Ss]tep\s+(\d+)")
LOSS_RE = re.compile(r"([A-Za-z][A-Za-z ]*?) Loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")


def parse(path):
    steps, series = [], {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            losses = LOSS_RE.findall(line)
            if not m or not losses:
                continue
            step = int(m.group(1))
            steps.append(step)
            row = {name.strip(): float(v) for name, v in losses}
            for name in set(series) | set(row):
                series.setdefault(name, [None] * (len(steps) - 1))
                series[name].append(row.get(name))
    return steps, series


def plot(steps, series, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if "Total" in series:
        axes[0].plot(steps, series["Total"], color="tab:blue")
    axes[0].set_title(title + ": total loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    for name, vals in series.items():
        if name == "Total":
            continue
        pts = [(s, v) for s, v in zip(steps, vals) if v is not None]
        if pts:
            axes[1].plot(*zip(*pts), label=name, lw=1.2)
    axes[1].set_title(title + ": loss components")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(name, steps, series):
    tot = series.get("Total")
    if not tot or not steps:
        print(name, ": no data parsed")
        return None
    pts = [(s, v) for s, v in zip(steps, tot) if v is not None]
    best = min(pts, key=lambda x: x[1])
    print("%s: %d points, first %.4f (step %d), final %.4f (step %d), best %.4f (step %d)"
          % (name, len(pts), pts[0][1], pts[0][0], pts[-1][1], pts[-1][0], best[1], best[0]))
    return pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ts, tser = parse(a.train)
    vs, vser = parse(a.val)
    plot(ts, tser, (a.name + " training").strip(), os.path.join(a.out, "training_loss.png"))
    plot(vs, vser, (a.name + " validation").strip(), os.path.join(a.out, "validation_loss.png"))
    summary("train", ts, tser)
    summary("val", vs, vser)
    print("Saved training_loss.png and validation_loss.png in", a.out)
