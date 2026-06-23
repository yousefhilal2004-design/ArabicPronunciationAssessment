"""
audio_utils.py
==============
Preprocessing pipeline per project spec:
  "Apply silence removal, normalization, framing, and windowing."
"""

import librosa
import numpy as np
import soundfile as sf


def load_audio(file_path, sr=16000):
    """Load audio file with specified sample rate."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def remove_silence(y, top_db=20):
    """Remove silence from beginning and end of audio."""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def normalize_audio(y):
    """Normalize audio to [-1, 1] range (peak normalization)."""
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val


def preprocess_audio(file_path, sr=16000):
    """Complete preprocessing pipeline: load -> silence removal -> normalize."""
    y, sr = load_audio(file_path, sr)
    if y is None:
        return None, None
    y = remove_silence(y)
    y = normalize_audio(y)
    return y, sr


def save_audio(y, sr, file_path):
    """Save audio to file."""
    sf.write(file_path, y, sr)


def apply_framing(y, sr, frame_size=0.025, frame_shift=0.010):
    """
    Apply framing to audio signal.
    Default: 25 ms frames, 10 ms shift -- standard short-time analysis values.
    """
    frame_length = int(frame_size * sr)
    hop_length = int(frame_shift * sr)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return frames, frame_length, hop_length


def apply_windowing(frames, window_type='hamming'):
    """Apply windowing to frames."""
    n = frames.shape[0]
    if window_type == 'hamming':
        window = np.hamming(n)
    elif window_type == 'hanning':
        window = np.hanning(n)
    elif window_type == 'blackman':
        window = np.blackman(n)
    else:
        window = np.ones(n)
    return frames * window[:, np.newaxis]