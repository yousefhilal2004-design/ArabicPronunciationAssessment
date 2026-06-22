"""
dataset.py
==========
Manages the Arabic pronunciation dataset.

Expected folder structure:
  dataset/
    speaker_1/
      arabi_1.wav
      daw_1.wav
      ...
    speaker_2/
      arabi_2.wav
      ...
"""

import os
import glob
import numpy as np

from audio_utils import preprocess_audio, load_audio
from feature_extraction import extract_all_features


class ArabicDataset:
    """Manages the Arabic pronunciation dataset."""

    # Canonical word keys
    TEST_WORDS = [
        'arabi', 'daw', 'ghorfa', 'hadeqa', 'khaled',
        'qalam', 'sadeeq', 'tareeq', 'thaletha', 'tharf'
    ]

    WORD_MAPPING = {w: w for w in TEST_WORDS}

    ARABIC_NAMES = {
        'arabi': 'عربي', 'daw': 'ضوء',     'ghorfa': 'غرفة',
        'hadeqa': 'حديقة', 'khaled': 'خالد', 'qalam': 'قلم',
        'sadeeq': 'صديق',  'tareeq': 'طريق', 'thaletha': 'ثلاثة',
        'tharf': 'ظرف',
    }

    ENGLISH_MEANING = {
        'arabi': 'Arabic', 'daw': 'Light',   'ghorfa': 'Room',
        'hadeqa': 'Garden', 'khaled': 'Khaled (name)', 'qalam': 'Pen',
        'sadeeq': 'Friend', 'tareeq': 'Road', 'thaletha': 'Three',
        'tharf': 'Envelope',
    }

    WORD_SOUNDS = {
        'arabi':    ['ع', 'ر', 'ب', 'ي'],
        'daw':      ['ض', 'و', 'ء'],
        'ghorfa':   ['غ', 'ر', 'ف', 'ة'],
        'hadeqa':   ['ح', 'د', 'ي', 'ق', 'ة'],
        'khaled':   ['خ', 'ل', 'د'],
        'qalam':    ['ق', 'ل', 'م'],
        'sadeeq':   ['ص', 'د', 'ي', 'ق'],
        'tareeq':   ['ط', 'ر', 'ي', 'ق'],
        'thaletha': ['ث', 'ل', 'ث', 'ة'],
        'tharf':    ['ظ', 'ر', 'ف'],
    }

    SOUND_TYPES = {
        'emphatic':    ['ط', 'ض', 'ص', 'ظ', 'ق'],
        'pharyngeal':  ['ح', 'ع'],
        'uvular':      ['خ', 'غ'],
        'regular':     ['ب', 'ت', 'ث', 'ج', 'د', 'ذ', 'ر', 'ز', 'س',
                        'ش', 'ف', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي'],
    }

    # ------------------------------------------------------------------
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.speakers     = []
        self.audio_files  = {}   # {speaker: {word: path}}
        self.features     = {}   # {speaker: {word: features}}
        self._load_dataset()

    # ------------------------------------------------------------------
    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            print(f"⚠️  Dataset path '{self.dataset_path}' not found.")
            return

        speaker_dirs = sorted(glob.glob(
            os.path.join(self.dataset_path, "speaker_*")
        ))

        for speaker_dir in speaker_dirs:
            speaker = os.path.basename(speaker_dir)
            self.speakers.append(speaker)
            self.audio_files[speaker] = {}

            for audio_file in glob.glob(os.path.join(speaker_dir, "*.wav")):
                filename = os.path.splitext(os.path.basename(audio_file))[0]
                # filename format: <word>_<number>  e.g. arabi_1
                word = filename.split('_')[0]

                if word in self.WORD_MAPPING:
                    self.audio_files[speaker][word] = audio_file
                else:
                    # Fallback: prefix match
                    for known_word in self.TEST_WORDS:
                        if filename.startswith(known_word):
                            self.audio_files[speaker][known_word] = audio_file
                            break

        print(f"\n📁 Dataset loaded: {self.dataset_path}")
        print(f"👥 Speakers found: {len(self.speakers)}")
        for sp in self.speakers:
            print(f"  • {sp}: {len(self.audio_files[sp])} files")

        self._verify_dataset()

    def _verify_dataset(self):
        missing = {}
        for sp in self.speakers:
            m = [w for w in self.TEST_WORDS if w not in self.audio_files[sp]]
            if m:
                missing[sp] = m

        if missing:
            print("\n⚠️  Missing audio files:")
            for sp, words in missing.items():
                print(f"  • {sp}: {', '.join(words)}")
        else:
            print("✅ All speakers have all 10 words.")

    # ------------------------------------------------------------------
    # Data accessors
    # ------------------------------------------------------------------

    def get_speaker_files(self, speaker):
        return self.audio_files.get(speaker, {})

    def get_word_files(self, word):
        return {
            sp: self.audio_files[sp][word]
            for sp in self.speakers
            if word in self.audio_files.get(sp, {})
        }

    def get_file_path(self, speaker, word):
        return self.audio_files.get(speaker, {}).get(word, None)

    def get_speaker_count(self):
        return len(self.speakers)

    def get_word_count(self):
        return len(self.TEST_WORDS)

    def get_test_words(self):
        return self.TEST_WORDS

    def get_arabic_name(self, word):
        return self.ARABIC_NAMES.get(word, word)

    def get_english_meaning(self, word):
        return self.ENGLISH_MEANING.get(word, word)

    def get_word_sounds(self, word):
        return self.WORD_SOUNDS.get(word, [])

    def get_sound_type(self, sound):
        for stype, sounds in self.SOUND_TYPES.items():
            if sound in sounds:
                return stype
        return 'unknown'

    def get_word_sound_types(self, word):
        return {s: self.get_sound_type(s) for s in self.get_word_sounds(word)}

    def get_arabic_sounds(self):
        return self.SOUND_TYPES

    def get_sound_examples(self, sound):
        return [w for w, sounds in self.WORD_SOUNDS.items() if sound in sounds]

    def get_speaker_word_count(self, speaker):
        return len(self.audio_files.get(speaker, {}))

    def get_all_speaker_data(self):
        return {
            sp: {w: self.get_file_path(sp, w) for w in self.TEST_WORDS
                 if self.get_file_path(sp, w)}
            for sp in self.speakers
        }

    def get_all_word_data(self):
        return {
            w: {sp: self.get_file_path(sp, w) for sp in self.speakers
                if self.get_file_path(sp, w)}
            for w in self.TEST_WORDS
        }

    def get_word_info(self):
        return {
            w: {
                'arabic':      self.get_arabic_name(w),
                'english':     self.get_english_meaning(w),
                'sounds':      self.get_word_sounds(w),
                'sound_types': self.get_word_sound_types(w),
                'speakers':    list(self.get_word_files(w).keys()),
            }
            for w in self.TEST_WORDS
        }

    # ------------------------------------------------------------------
    # Feature extraction (cached)
    # ------------------------------------------------------------------

    def extract_features_for_speaker(self, speaker):
        if speaker not in self.audio_files:
            return {}
        if speaker not in self.features:
            self.features[speaker] = {}
            for word, path in self.audio_files[speaker].items():
                y, sr = preprocess_audio(path)
                if y is not None:
                    self.features[speaker][word] = extract_all_features(path, y, sr)
        return self.features[speaker]

    def extract_features_for_word(self, word):
        result = {}
        for sp in self.speakers:
            path = self.get_file_path(sp, word)
            if path:
                y, sr = preprocess_audio(path)
                if y is not None:
                    result[sp] = extract_all_features(path, y, sr)
        return result

    # ------------------------------------------------------------------
    # Info printing
    # ------------------------------------------------------------------

    def print_dataset_info(self):
        print("\n" + "=" * 70)
        print("ARABIC PRONUNCIATION DATASET INFORMATION")
        print("=" * 70)
        print(f"\n📁 Path:    {self.dataset_path}")
        print(f"👥 Speakers: {self.get_speaker_count()}")
        print(f"📝 Words:    {self.get_word_count()}")
        print(f"📊 Total:    {self.get_speaker_count() * self.get_word_count()} files")

        print("\n📋 Word List:")
        print("-" * 70)
        print(f"{'Word':<15} {'Arabic':<10} {'English':<15} {'Sounds':<25}")
        print("-" * 70)
        for w in self.TEST_WORDS:
            sounds = ', '.join(self.get_word_sounds(w))
            print(f"{w:<15} {self.get_arabic_name(w):<10} "
                  f"{self.get_english_meaning(w):<15} {sounds:<25}")

        print("\n🔊 Sound Types:")
        print("-" * 70)
        for stype, sounds in self.SOUND_TYPES.items():
            print(f"  {stype.upper():<14}: {', '.join(sounds)}")

        print("\n👥 Speaker Files:")
        print("-" * 70)
        for sp in self.speakers:
            words = list(self.audio_files[sp].keys())
            print(f"  {sp}: {', '.join(words)}")
        print("=" * 70)