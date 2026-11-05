""" Symbol table for the Telugu model (character-level phones).

Adapted from the reference FastSpeech2 symbols.py: the English ARPAbet and
pinyin symbols are replaced by the Telugu Unicode block (U+0C00-U+0C7F) and
ASCII digits. Each Telugu character is one phone, matching the grapheme-based
dictionary used to train the Telugu MFA aligner. """

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@spn", "@sil"]

# "@" prefix marks phone symbols, as in the reference code (text_to_sequence
# turns everything inside {...} into "@<symbol>")
_telugu = ["@" + chr(c) for c in range(0x0C00, 0x0C80)]
_digits = ["@" + d for d in "0123456789"]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + _telugu + _digits + _silences
