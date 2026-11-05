import json
import os
import shutil

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio
from text.symbols import symbols

# Our own split: raw_path/train/<speaker>/*.wav and raw_path/val/<speaker>/*.wav
SPLITS = ["train", "val"]


class Preprocessor:
    def __init__(self, config, limit=None):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.limit = limit  # smoke test: only process the first N clips of each split
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        # every phone label written to train.txt must exist in the symbol table,
        # otherwise text_to_sequence silently drops it and lengths stop matching
        self.symbol_set = set(symbols)

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def index_textgrids(self):
        """Map utterance basename -> TextGrid path, wherever MFA put it."""
        index = {}
        tg_root = os.path.join(self.out_dir, "TextGrid")
        for root, _, files in os.walk(tg_root):
            for f in files:
                if f.endswith(".TextGrid"):
                    index[f[: -len(".TextGrid")]] = os.path.join(root, f)
        return index

    def build_from_path(self):
        # start from clean feature folders so no stale files survive a re-run
        for sub in ["mel", "pitch", "energy", "duration"]:
            path = os.path.join(self.out_dir, sub)
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        tg_index = self.index_textgrids()
        print("TextGrid files found:", len(tg_index))

        speakers = sorted(
            {
                spk
                for split in SPLITS
                for spk in os.listdir(os.path.join(self.in_dir, split))
                if os.path.isdir(os.path.join(self.in_dir, split, spk))
            }
        )
        speaker_map = {spk: i for i, spk in enumerate(speakers)}

        print("Processing Data ...")
        out = {split: [] for split in SPLITS}
        n_frames = {split: 0 for split in SPLITS}
        skipped = []
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        for split in SPLITS:
            items = []
            split_dir = os.path.join(self.in_dir, split)
            for speaker in sorted(os.listdir(split_dir)):
                spk_dir = os.path.join(split_dir, speaker)
                if not os.path.isdir(spk_dir):
                    continue
                for wav_name in sorted(os.listdir(spk_dir)):
                    if wav_name.endswith(".wav"):
                        items.append((speaker, wav_name[: -len(".wav")]))
            if self.limit is not None:
                items = items[: self.limit]

            for speaker, basename in tqdm(items, desc=split):
                tg_path = tg_index.get(basename)
                if tg_path is None:
                    skipped.append("|".join([split, basename, "no TextGrid"]))
                    continue
                ret = self.process_utterance(split, speaker, basename, tg_path)
                if isinstance(ret, str):
                    skipped.append("|".join([split, basename, ret]))
                    continue
                info, pitch, energy, n = ret
                out[split].append(info)
                n_frames[split] += n
                # normalisation statistics come from the training split only
                if split == "train":
                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

        print("Computing statistic quantities ...")
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speaker_map))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        for split in SPLITS:
            print(
                "{}: {} clips kept, {:.2f} hours".format(
                    split,
                    len(out[split]),
                    n_frames[split] * self.hop_length / self.sampling_rate / 3600,
                )
            )
        print("Skipped clips:", len(skipped))

        with open(os.path.join(self.out_dir, "skipped.txt"), "w", encoding="utf-8") as f:
            for line in skipped:
                f.write(line + "\n")

        # Write metadata: train.txt / val.txt follow OUR split, not a random one
        for split in SPLITS:
            with open(
                os.path.join(self.out_dir, "{}.txt".format(split)),
                "w",
                encoding="utf-8",
            ) as f:
                for m in out[split]:
                    f.write(m + "\n")

        return out

    def process_utterance(self, split, speaker, basename, tg_path):
        """Returns a tuple on success, or a string describing why the clip was skipped."""
        wav_path = os.path.join(self.in_dir, split, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, split, speaker, "{}.lab".format(basename))

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        if len(phone) == 0 or start >= end:
            return "empty alignment"
        unknown = sorted({p for p in phone if "@" + p not in self.symbol_set})
        if unknown:
            return "unknown phones: " + " ".join(unknown)
        text = "{" + " ".join(phone) + "}"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return "no voiced frames"

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        if mel_spectrogram.shape[1] < sum(duration):
            return "audio shorter than alignment"
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        intervals = getattr(tier, "intervals", None)
        if intervals is None:
            intervals = tier._objects
        for t in intervals:
            s, e, p = t.start_time, t.end_time, t.text

            # MFA 3.x writes silence as an EMPTY label; the reference code only
            # knows sil/sp/spn, so map the empty label to "sp"
            if p == "":
                p = "sp"

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value
