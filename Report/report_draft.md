# Bilingual Text-to-Speech with FastSpeech 2: English and Telugu

## 2. Datasets and Preprocessing

### 2.1 English (Svarah)
- Source: ai4bharat/Svarah on Hugging Face. It is an Indian-accented English speech dataset built for ASR evaluation, with 6,656 clips (9.61 h) in a single `test` split and no speaker ID.
- Speaker handling: because no speaker ID exists, a speaker profile was built by concatenating the metadata columns (gender, age group, primary language, native state, native district, highest qualification, job category, occupation domain). This gives 115 profiles. A profile is only an approximation of a speaker and may merge different people with identical metadata. No profile has more than 0.41 h of audio, so a single-speaker model could not reach 5 h. The English model is therefore multi-speaker.
- Filtering: clips shorter than 1 s or longer than 20 s were removed, leaving 5,331 clips (8.68 h).
- Split: random seed 42. Clips with identical text (case-insensitive) were assigned to one side only, so no sentence appears in both train and validation. Validation was filled first to 1.00 h, then training clips were added to 5.00 h.
- Result: train = 3,040 clips (5.00 h); validation = 607 clips (1.00 h). All 115 profiles appear in train and 108 in validation. Overlap was checked in code (by clip index and by text).
- Audio: resampled to 22.05 kHz mono.

### 2.2 Telugu (IndicTTS Telugu)
- Source: SPRINGLab/IndicTTS_Telugu on Hugging Face: 8,576 clips (36.7 h), 48 kHz, in a single `train` split, with a `gender` column (0/1).
- Speaker choice: listening to clips showed gender 0 is a female voice and gender 1 a male voice. Gender 1 was chosen because it has 8.08 h of clips between 2 and 15 s, against 3.74 h for gender 0, and short clips suit FastSpeech 2 training. Gender 1 is assumed to be a single male speaker, supported by listening to 8 gender 1 clips (2 random, 6 at the extremes of estimated pitch), all male. An automated pitch check was inconclusive.
- Filtering: clips between 2 and 15 s (3,750 clips, 8.08 h).
- Split: same method as English (seed 42, identical sentences on one side only).
- Result: train = 2,315 clips (5.00 h); validation = 459 clips (1.00 h). No overlap by clip index or text (checked in code).
- Audio: resampled from 48 kHz to 22.05 kHz mono.

### 2.3 Training configuration
TODO: fill in once training is set up (batch size, steps, learning rate, vocoder).

### 2.4 Test sentences
The test sentences were used exactly as given in the assignment, including the irregular spellings "rhyth m" (E3) and "andaroo" (E4).