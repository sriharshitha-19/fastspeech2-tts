# Bilingual Text-to-Speech with FastSpeech 2: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models, one for Indian-accented English and one for Telugu, adapted from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence).
The full analysis is in `Report/Technical_Report.pdf`.

## Repository structure

```
project/
├── English/   train.py  preprocess.py  inference.py  config.yaml  checkpoints/  (plus the modified reference code)
├── Telugu/    train.py  preprocess.py  inference.py  config.yaml  checkpoints/  (plus the modified reference code)
├── Results/
│   ├── English/  training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
│   └── Telugu/   training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
├── Report/Technical_Report.pdf
├── plot_losses.py   patch_librosa.py   CHANGES.md
├── README.md
└── requirements.txt
```

## What was done

| | English | Telugu |
|---|---|---|
| Data | Svarah (Indian-accented, multi-speaker) | IndicTTS Telugu (one speaker) |
| Training / validation audio kept | 3.68 h (2,979 clips) / 0.74 h (599 clips) | 3.67 h (2,314 clips) / 0.74 h (459 clips) |
| Symbols | ARPAbet phones (lexicon + g2p_en) | Telugu characters (one symbol per Unicode character) |
| Steps trained | 15,629 of 30,000 planned | about 14,800 of 30,000 planned |
| Best / final validation loss | 3.32 (step 2,500) / 3.51 | 3.72 (step 1,000) / 4.01 |

Training was limited to a 120-minute budget on Kaggle (GPU T4 x2), so both models stop at about half of the planned steps.
Both over-fit the small training sets. In our listening notes the Telugu model gives intelligible speech for T2a and T2b,
while the English output is broken noise with no clear words. These limitations and ideas for improvement are discussed in the report.

The changes made to the reference code are listed in `CHANGES.md` and in section 1 of the report.

### Test sentences
`English/test_sentences.txt` and `Telugu/test_sentences.txt` hold the five test sentences of each language (format `ID|text`).
The English sentences are exactly as given in the assignment (including "rhyth m" and "andaroo").
The Telugu sentences in the assignment PDF were damaged by copy-and-paste, so they were retyped as correct Telugu
(for example T3 `అడవిలో` and T2b `పట్టుచీరను`). This is also noted in the report.

## Checkpoints

The trained model files (weights only, about 138 MB each) are too large for a normal Git repository:

- English checkpoint (step 15,629): https://drive.google.com/file/d/1cON9Bd_YhnwyvuGb-YEWrNWEIMqmGCwJ/view?usp=sharing
- Telugu checkpoint (about step 14,800): https://drive.google.com/file/d/1refaq0tPguNQBjVEZFJv6JqwUjvdxNZT/view?usp=sharing

To use a checkpoint, save it as `English/checkpoints/15629.pth.tar` (English) or `Telugu/checkpoints/<step>.pth.tar` (Telugu).
`inference.py` picks the file with the highest step number.

## Setup

```
pip install -r requirements.txt
```

The code was run on Kaggle (Python 3.13, a recent PyTorch, GPU T4 x2). `patch_librosa.py` adapts the reference code to current
librosa and torch versions. `g2p_en` downloads some NLTK data on first use (needs internet).
Download the pretrained universal HiFi-GAN generator from the reference repository and put it in `<Language>/hifigan/generator_universal.pth.tar`.

## Reproduction (run inside the language folder, shown for English)

1. **Prepare the data.** Export the audio as 22,050 Hz mono WAV files into `raw_data/<dataset>/train/` and `raw_data/<dataset>/val/`
   (dataset names: `Svarah`, `IndicTTS_Telugu`; see `config.yaml`), and place the forced-alignment TextGrids in
   `preprocessed_data/<dataset>/TextGrid/`. The Kaggle notebook `fs2_kaggle.ipynb` automates these steps.
2. **Preprocess** (mel, pitch, energy, durations, `stats.json`):
   ```
   cd English
   python preprocess.py config.yaml
   python preprocess.py config.yaml --limit 20      # optional quick test on a few clips
   ```
3. **Train** (stops cleanly after the given number of minutes, keeps the last three checkpoints):
   ```
   python train.py -c config.yaml --max_minutes 120
   ```
   Losses are written to `output/log/train/log.txt` and `output/log/val/log.txt`.
4. **Synthesise the test sentences and plot duration, pitch and energy** (needs a checkpoint in `checkpoints/` and `stats.json`
   in the preprocessed data folder):
   ```
   python inference.py --config config.yaml --sentences test_sentences.txt --out_dir ../Results/English --speaker_id 0
   ```
5. **Loss plots** (from the project root):
   ```
   python plot_losses.py --train <log_dir>/train/log.txt --val <log_dir>/val/log.txt --out Results/English --name English
   ```

For Telugu, run the same commands inside `Telugu/` (no `--speaker_id`), with `--out_dir ../Results/Telugu` and `--name Telugu`.

## Acknowledgements and licences

- Model code adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence), which implements
  [FastSpeech 2](https://arxiv.org/abs/2006.04558).
- Vocoder: the universal HiFi-GAN generator distributed with the reference repository.
- Data: Svarah (English) and IndicTTS Telugu; please follow the licences of the original datasets.
