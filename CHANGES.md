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