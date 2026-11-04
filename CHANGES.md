# Changes log

## English (Svarah)
- Svarah has no speaker ID, so speaker profiles were built from metadata columns.
- Clips filtered to 1 to 20 s. Split by utterance with identical sentences kept on one side (seed 42): 3,040 train clips (5.00 h), 607 validation clips (1.00 h).
- Audio resampled to 22.05 kHz mono. Model is multi-speaker (115 profiles).

## Telugu (IndicTTS)
- gender 0 = female, gender 1 = male (verified by listening). Gender 1 chosen: 8.08 h of 2 to 15 s clips vs 3.74 h for gender 0.
- Gender 1 assumed to be a single male speaker: 8 clips checked (2 random, 6 at pitch extremes), all male. Automated pitch check was inconclusive.
- Clips filtered to 2 to 15 s. Same split method (seed 42): 2,315 train clips (5.00 h), 459 validation clips (1.00 h).
- Audio resampled from 48 kHz to 22.05 kHz.

## 2026-11-03 (continued)

- Cloned the reference repo ming024/FastSpeech2 into the Colab environment.
- The repo's requirements.txt pins old library versions; installed current versions of the dependencies instead. Because of the version changes, some reference code may need patching. Each patch will be logged here.
- Installed Montreal Forced Aligner 3.4.3 via micromamba (conda-forge) in a separate Python 3.11 environment instead of pip.

## 2026-11-04
- English alignment: 3,643 of 3,647 clips aligned; 4 failed and were excluded. Alignments saved to Google Drive.
- Copied the reference FastSpeech2 code into English/ and Telugu/, leaving out the pretrained vocoder weights (.zip), demo files, images and the reference repo's preprocessed_data.
- English/config.yaml: shortened training (total_step 30000 instead of 900000, warm_up_step 2000, anneal_steps 20000/25000/28000), validation every 500 steps for the loss plot, checkpoint every 1000 steps because Colab sessions drop.
- First Telugu aligner run (default MFA training recipe, 2 CPU jobs) was stopped after about 90 minutes without producing alignments or a saved model; rerunning with a shorter monophone and triphone recipe.