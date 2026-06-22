import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract MFCC features (13 coefficients as required by project spec).
    NOT normalised — raw values are used for cosine similarity comparison.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc  # shape: (n_mfcc, frames)


def extract_pitch(file_path):
    """
    Extract mean pitch (F0) in Hz using Parselmouth.
    Returns 0 if unvoiced / extraction fails.
    """
    try:
        sound  = parselmouth.Sound(file_path)
        pitch  = sound.to_pitch()
        values = pitch.selected_array['frequency']
        voiced = values[values > 0]
        if len(voiced) == 0:
            return 0.0
        return float(np.mean(voiced))
    except Exception:
        return 0.0


def extract_pitch_contour(file_path):
    """Extract full pitch contour array (Hz per frame)."""
    try:
        sound  = parselmouth.Sound(file_path)
        pitch  = sound.to_pitch()
        values = pitch.selected_array['frequency']
        return values.astype(float)
    except Exception:
        return np.array([])


def extract_duration(y, sr):
    """Return audio duration in seconds."""
    return len(y) / sr


def extract_energy(y, frame_length=2048, hop_length=512):
    """Short-time energy per frame."""
    energy = np.array([
        np.sum(y[i:i + frame_length] ** 2)
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    return energy


def extract_total_energy(y):
    """Mean squared energy over entire signal."""
    return float(np.mean(y ** 2))


def extract_zcr(y, frame_length=2048, hop_length=512):
    """Mean zero-crossing rate."""
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )
    return float(np.mean(zcr))


def extract_formants(file_path, num_formants=3):
    """
    Extract F1, F2, F3 at the midpoint of the utterance using Parselmouth.
    Returns a list of floats; 0 if a formant cannot be found.
    """
    try:
        sound   = parselmouth.Sound(file_path)
        formant = sound.to_formant_burg(
            max_number_of_formants=5.0,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0
        )
        midpoint = sound.duration / 2.0
        result = []
        for i in range(1, num_formants + 1):
            try:
                val = formant.get_value_at_time(i, midpoint)
                result.append(float(val) if (val and not np.isnan(val)) else 0.0)
            except Exception:
                result.append(0.0)
        return result
    except Exception:
        return [0.0] * num_formants


def extract_formant_contour(file_path, num_formants=3):
    """Extract F1–F3 contours over time (shape: num_formants × time_steps)."""
    try:
        sound   = parselmouth.Sound(file_path)
        formant = sound.to_formant_burg()
        times   = np.arange(0, sound.duration, 0.01)
        contours = []
        for i in range(1, num_formants + 1):
            contour = []
            for t in times:
                try:
                    val = formant.get_value_at_time(i, t)
                    contour.append(float(val) if (val and not np.isnan(val)) else 0.0)
                except Exception:
                    contour.append(0.0)
            contours.append(contour)
        return np.array(contours)
    except Exception:
        return np.zeros((num_formants, 1))


def extract_all_features(file_path, y, sr):
    """Extract and return all features as a dictionary."""
    return {
        'mfcc':             extract_mfcc(y, sr),
        'pitch':            extract_pitch(file_path),
        'pitch_contour':    extract_pitch_contour(file_path),
        'duration':         extract_duration(y, sr),
        'energy':           extract_total_energy(y),
        'energy_contour':   extract_energy(y),
        'zcr':              extract_zcr(y),
        'formants':         extract_formants(file_path),
        'formant_contours': extract_formant_contour(file_path),
    }