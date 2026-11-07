"""FastSpeech 2 inference for English and Telugu: synthesize the test sentences
and save audio plus variance-adaptor plots (predicted duration, pitch/F0, energy).

Adapted from ming024/FastSpeech2 (MIT licence), synthesize.py.
The same file is used in English/ and Telugu/. Run it from inside the language
folder (config paths are relative), for example:

  cd English
  python inference.py --config config.yaml --sentences test_sentences.txt \
      --out_dir ../Results/English --speaker_id 0

  cd Telugu
  python inference.py --config config.yaml --sentences test_sentences.txt \
      --out_dir ../Results/Telugu

Sentence file format: one sentence per line as  ID|text  (lines starting with #
are ignored). The checkpoint step defaults to the latest one in ckpt_path.
"""
import os
import re
import csv
import glob
import string
import argparse
import unicodedata

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.io import wavfile

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Telugu: insert a short pause symbol ("sp") at every space? Training phones only
# contain "sp" where the aligner found a pause, so the default is False.
WORD_SP = False


# ----------------------------------------------------------------------------
# text front ends
# ----------------------------------------------------------------------------
def telugu_phones(text):
    """Telugu text -> list of phones. Each Telugu character is one phone (as in
    training); punctuation becomes "sp"; digits stay as digit symbols."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ / ZWJ
    text = text.strip().rstrip(string.punctuation + "\u0964\u0965")
    phones = []
    for ch in text:
        if 0x0C00 <= ord(ch) <= 0x0C7F or ch in "0123456789":
            phones.append(ch)
        elif ch.isspace():
            if WORD_SP and phones and phones[-1] != "sp":
                phones.append("sp")
        else:  # punctuation or any other character
            if phones and phones[-1] != "sp":
                phones.append("sp")
    return phones


def preprocess_telugu(text, preprocess_config):
    phones = telugu_phones(text)
    if not phones:
        raise ValueError("Empty Telugu phone sequence for: %r" % text)
    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    return np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )


def encode_text(text, preprocess_config):
    """Raw text -> array of symbol ids, chosen by the language in the config."""
    lang = preprocess_config["preprocessing"]["text"]["language"]
    if lang == "en":
        import synthesize as base  # reference English front end (lexicon + g2p_en)

        return base.preprocess_english(text, preprocess_config)
    if lang == "te":
        return preprocess_telugu(text, preprocess_config)
    raise NotImplementedError("No text front end for language '%s'" % lang)


def id_to_label_fn():
    """Map symbol ids back to readable phoneme/character labels for the plots."""
    try:
        from text.symbols import symbols

        def f(i):
            s = symbols[i]
            return s[1:] if s.startswith("@") else s

        return f
    except Exception:
        return lambda i: str(i)


def pick_font():
    """Pick a font that can draw Telugu if one is installed."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "Noto Sans Telugu",
        "Nirmala UI",
        "Gautami",
        "Lohit Telugu",
        "Vani",
        "DejaVu Sans",
    ]:
        if name in names:
            matplotlib.rcParams["font.family"] = ["DejaVu Sans",name]
            return name
    return None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def latest_step(ckpt_dir):
    steps = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.pth.tar")):
        m = re.match(r"(\d+)\.pth\.tar$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError("No <step>.pth.tar checkpoint found in %s" % ckpt_dir)
    return max(steps)


def read_sentences(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            sid, text = line.split("|", 1)
            out.append((sid.strip(), text.strip()))
    return out


def load_stats(preprocess_config):
    import json

    path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    with open(path) as f:
        return json.load(f)


def denormalize(values, feat_cfg, stats_entry):
    # stats entry = [min, max, mean, std]
    if feat_cfg.get("normalization", False):
        return values * stats_entry[3] + stats_entry[2]
    return values


def expand(values, durations):
    return np.repeat(values, durations.astype(int))


def plot_duration(dur_ms, labels, title, path):
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 4))
    ax.bar(np.arange(len(dur_ms)), dur_ms, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Predicted duration (ms)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curve(values, d_frames, labels, sec_per_frame, title, ylabel, color, path):
    t = np.arange(len(values)) * sec_per_frame
    bounds = np.cumsum(d_frames) * sec_per_frame
    centers = bounds - d_frames * sec_per_frame / 2.0
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(labels)), 4.2))
    ax.plot(t, values, lw=1.5, color=color)
    for b in bounds:
        ax.axvline(b, color="gray", lw=0.3, alpha=0.5)
    top = ax.secondary_xaxis("top")
    top.set_xticks(centers)
    top.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(t[-1], bounds[-1]) if len(t) else 1)
    ax.set_title(title, pad=40)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main synthesis
# ----------------------------------------------------------------------------
def synth_one(sid, text, model, vocoder, configs, args, dirs, label_fn, stats):
    pre, mdl, _ = configs
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    hop = pre["preprocessing"]["stft"]["hop_length"]
    sec_per_frame = hop / float(sr)

    seq = np.array(encode_text(text, pre))
    texts = np.array([seq])
    src_lens = np.array([len(seq)])
    speakers = np.array([args.speaker_id])
    batch = ([sid], [text[:100]], speakers, texts, src_lens, max(src_lens))
    batch = to_device(batch, device)

    with torch.no_grad():
        out = model(
            *(batch[2:]),
            p_control=args.pitch_control,
            e_control=args.energy_control,
            d_control=args.duration_control
        )
        # out: (mel, postnet_mel, pitch, energy, log_dur, dur_rounded,
        #       src_masks, mel_masks, src_lens, mel_lens)
        mel_len = int(out[9][0])
        src_len = int(out[8][0])
        mel = out[1][:, :mel_len].transpose(1, 2)
        wav = vocoder_infer(mel, vocoder, mdl, pre, lengths=[mel_len * hop])[0]

    wavfile.write(os.path.join(dirs["audio"], "%s.wav" % sid), sr, wav)

    d = out[5][0, :src_len].cpu().numpy().astype(int)
    d = np.maximum(d, 0)
    p = out[2][0].cpu().numpy()
    e = out[3][0].cpu().numpy()

    if pre["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        p = expand(p[:src_len], d)
    if pre["preprocessing"]["energy"]["feature"] == "phoneme_level":
        e = expand(e[:src_len], d)
    p, e = p[:mel_len], e[:mel_len]

    p = denormalize(p, pre["preprocessing"]["pitch"], stats["pitch"])
    e = denormalize(e, pre["preprocessing"]["energy"], stats["energy"])

    labels = [label_fn(int(i)) for i in seq]
    dur_ms = d * sec_per_frame * 1000.0

    plot_duration(dur_ms, labels, "%s: predicted duration per symbol" % sid,
                  os.path.join(dirs["duration"], "%s.png" % sid))
    plot_curve(p, d, labels, sec_per_frame, "%s: predicted pitch (F0)" % sid,
               "F0 (Hz)", "tab:red", os.path.join(dirs["pitch"], "%s.png" % sid))
    plot_curve(e, d, labels, sec_per_frame, "%s: predicted energy" % sid,
               "Energy", "tab:green", os.path.join(dirs["energy"], "%s.png" % sid))

    return {
        "id": sid,
        "text": text,
        "n_symbols": len(seq),
        "audio_seconds": round(mel_len * sec_per_frame, 2),
        "mean_symbol_ms": round(float(dur_ms.mean()), 1),
        "max_symbol_ms": round(float(dur_ms.max()), 1),
        "f0_mean_hz": round(float(np.mean(p)), 1),
        "f0_min_hz": round(float(np.min(p)), 1),
        "f0_max_hz": round(float(np.max(p)), 1),
        "energy_mean": round(float(np.mean(e)), 3),
        "energy_max": round(float(np.max(e)), 3),
    }


def load_configs(args):
    if args.config:
        cfg = yaml.load(open(args.config, encoding="utf-8"), Loader=yaml.FullLoader)
        return cfg["preprocess"], cfg["model"], cfg["train"]
    if not (args.preprocess_config and args.model_config and args.train_config):
        raise SystemExit("Give --config, or all of -p -m -t")
    return tuple(
        yaml.load(open(p, encoding="utf-8"), Loader=yaml.FullLoader)
        for p in (args.preprocess_config, args.model_config, args.train_config)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="single config.yaml with preprocess/model/train sections")
    parser.add_argument("-p", "--preprocess_config", type=str, default=None)
    parser.add_argument("-m", "--model_config", type=str, default=None)
    parser.add_argument("-t", "--train_config", type=str, default=None)
    parser.add_argument("--restore_step", type=int, default=None,
                        help="checkpoint step; default = latest in ckpt_path")
    parser.add_argument("--sentences", type=str, required=True,
                        help="file with lines  ID|text")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="e.g. ../Results/English")
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    parser.add_argument("--word_pause", action="store_true",
                        help="Telugu only: put a short pause between words")
    args = parser.parse_args()
    WORD_SP = args.word_pause

    configs = load_configs(args)
    preprocess_config, model_config, train_config = configs

    if args.restore_step is None:
        args.restore_step = latest_step(train_config["path"]["ckpt_path"])
    print("Using checkpoint step:", args.restore_step)

    pick_font()
    label_fn = id_to_label_fn()
    stats = load_stats(preprocess_config)

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    dirs = {
        "audio": os.path.join(args.out_dir, "test_generated_audios"),
        "duration": os.path.join(args.out_dir, "variance_adaptor_plots", "duration"),
        "pitch": os.path.join(args.out_dir, "variance_adaptor_plots", "pitch"),
        "energy": os.path.join(args.out_dir, "variance_adaptor_plots", "energy"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    rows = []
    for sid, text in read_sentences(args.sentences):
        print("Synthesizing", sid, ":", text)
        rows.append(synth_one(sid, text, model, vocoder, configs, args, dirs,
                              label_fn, stats))

    if rows:
        with open(os.path.join(args.out_dir, "inference_summary.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("Done. Results in", args.out_dir)
