"""Run once from the project root:  py patch_librosa.py

The reference FastSpeech2 code was written for old library versions. This patches
three incompatibilities in English/ and Telugu/ and prints what it changed:
  1. librosa.util.pad_center(x, n)       -> pad_center(x, size=n)
  2. librosa.filters.mel(sr, n_fft, ...) -> keyword arguments
  3. torch.load(path)                    -> torch.load(path, weights_only=False)
It is safe to run twice (already-patched code is left alone).
"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = ["audio/stft.py", "audio/audio_processing.py", "utils/model.py"]

PAD = re.compile(r"pad_center\(\s*(\w+)\s*,\s*(\w+)\s*\)")
MEL = re.compile(
    r"librosa_mel_fn\(\s*sampling_rate\s*,\s*filter_length\s*,\s*n_mel_channels\s*,\s*mel_fmin\s*,\s*mel_fmax\s*\)"
)
LOAD = re.compile(r"torch\.load\(([^()]*)\)")


def fix_load(m):
    if "weights_only" in m.group(1):
        return m.group(0)
    return "torch.load({}, weights_only=False)".format(m.group(1))


for lang in ["English", "Telugu"]:
    for rel in FILES:
        path = os.path.join(ROOT, lang, rel)
        if not os.path.exists(path):
            print("MISSING  ", path)
            continue
        text = open(path, encoding="utf-8").read()
        new = PAD.sub(r"pad_center(\1, size=\2)", text)
        new = MEL.sub(
            "librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, "
            "n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)",
            new,
        )
        new = LOAD.sub(fix_load, new)
        if new != text:
            open(path, "w", encoding="utf-8").write(new)
            print("PATCHED  ", os.path.join(lang, rel))
        else:
            print("no change", os.path.join(lang, rel))
