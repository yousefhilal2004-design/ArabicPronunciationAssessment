"""
comparison.py
=============
Rule-based comparison functions per project spec:
  "Rule-based methods: Euclidean distance, Cosine similarity"
  (DTW added for sequence alignment of variable-length MFCC matrices,
   listed in the required software: dtw-python)

All functions return a similarity score in [0, 100].

IMPORTANT: this version computes similarity directly from the acoustic
distance/correlation with NO manual scale factors chosen to push results
into a particular target range. The actual score distribution should be
measured empirically by running experiments.py on your dataset, then
reported honestly in the report -- not tuned to match a expected number.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


# ---------------------------------------------------------------------------
# DTW distance (manual implementation, as allowed by "dtw-python" spec --
# this gives full visibility into the algorithm for the report)
# ---------------------------------------------------------------------------

def dtw_distance(seq1, seq2):
    """
    Classic DTW with Euclidean local cost, O(n*m) dynamic programming.
    seq1, seq2: arrays of shape (frames, features).
    Returns the (unnormalized) DTW distance between the two sequences.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]


def compare_mfcc(mfcc1, mfcc2, method='dtw'):
    """
    Compare two MFCC matrices using DTW (sequence-aware) or cosine (fallback).

    DTW path:
      1. z-score normalize each sequence independently (removes overall
         amplitude/scale differences that aren't about pronunciation shape).
      2. Compute DTW distance.
      3. Normalize distance by the average sequence length (path length
         scales roughly with min(n, m), so this keeps the metric comparable
         across word pairs of different durations).
      4. Convert distance -> similarity with exponential decay.

    No scale factor here is hand-picked to hit a target score range --
    decay rate is fixed at a standard value (1.0) for reproducibility.
    """
    if mfcc1 is None or mfcc2 is None:
        return 0.0
    if mfcc1.shape[1] == 0 or mfcc2.shape[1] == 0:
        return 0.0

    if method == 'dtw':
        try:
            seq1 = mfcc1.T  # (frames, n_mfcc)
            seq2 = mfcc2.T

            # Per-sequence z-score normalization
            seq1 = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-10)
            seq2 = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-10)

            distance = dtw_distance(seq1, seq2)

            avg_len = (mfcc1.shape[1] + mfcc2.shape[1]) / 2.0
            normalized_distance = distance / (avg_len + 1e-10)

            # Fixed decay constant -- not tuned per-experiment
            similarity = 100.0 * np.exp(-normalized_distance)
            return float(np.clip(similarity, 0.0, 100.0))

        except Exception as e:
            print(f"DTW error, using cosine fallback: {e}")
            return compare_mfcc_cosine(mfcc1, mfcc2)

    return compare_mfcc_cosine(mfcc1, mfcc2)


def compare_mfcc_cosine(mfcc1, mfcc2):
    """Fallback: cosine similarity between mean MFCC vectors."""
    mean1 = np.mean(mfcc1, axis=1).reshape(1, -1)
    mean2 = np.mean(mfcc2, axis=1).reshape(1, -1)
    sim = float(cosine_similarity(mean1, mean2)[0][0])
    sim = np.clip(sim, -1.0, 1.0)
    score = (sim + 1.0) / 2.0 * 100.0
    return float(np.clip(score, 0.0, 100.0))


def compare_mfcc_euclidean(mfcc1, mfcc2):
    """
    Direct Euclidean-distance comparison of mean MFCC vectors, as
    explicitly named in the project spec ("Euclidean distance").
    Converted to a [0,100] similarity via exponential decay.
    """
    mean1 = np.mean(mfcc1, axis=1)
    mean2 = np.mean(mfcc2, axis=1)
    dist = euclidean(mean1, mean2)
    # Normalize by the scale of the reference vector's norm
    norm = np.linalg.norm(mean1) + 1e-10
    normalized_dist = dist / norm
    similarity = 100.0 * np.exp(-normalized_dist)
    return float(np.clip(similarity, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Pitch (scalar mean F0)
# ---------------------------------------------------------------------------

def compare_pitch(p1, p2):
    """Relative difference between mean pitch values (Hz)."""
    if p1 == 0 and p2 == 0:
        return 100.0
    if p1 == 0 or p2 == 0:
        return 0.0
    diff_ratio = abs(p1 - p2) / max(p1, p2, 1.0)
    score = 100.0 * (1.0 - diff_ratio)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Pitch contour (full array) -- used for richer prosody comparison
# ---------------------------------------------------------------------------

def compare_pitch_contour(pitch1, pitch2):
    """Pearson correlation between voiced-frame pitch contours."""
    if len(pitch1) == 0 or len(pitch2) == 0:
        return 0.0

    p1 = pitch1[pitch1 > 0]
    p2 = pitch2[pitch2 > 0]
    if len(p1) < 3 or len(p2) < 3:
        return 0.0

    min_len = min(len(p1), len(p2))
    p1, p2 = p1[:min_len], p2[:min_len]

    try:
        corr = np.corrcoef(p1, p2)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        score = (corr + 1.0) / 2.0 * 100.0
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def compare_duration(d1, d2):
    """Relative difference between utterance durations."""
    if d1 == 0 and d2 == 0:
        return 100.0
    if d1 == 0 or d2 == 0:
        return 0.0
    diff_ratio = abs(d1 - d2) / max(d1, d2, 1.0)
    score = 100.0 * (1.0 - diff_ratio)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Formants
# ---------------------------------------------------------------------------

def compare_formants(f1, f2):
    """Average relative difference across F1, F2, F3."""
    f1 = np.array(f1, dtype=float)
    f2 = np.array(f2, dtype=float)

    f1 = np.where(f1 == 0, 500.0, f1)
    f2 = np.where(f2 == 0, 500.0, f2)

    n = min(len(f1), len(f2))
    if n == 0:
        return 0.0

    diffs = [abs(f1[i] - f2[i]) / max(f1[i], f2[i], 1.0) for i in range(n)]
    avg_diff = float(np.mean(diffs))
    score = 100.0 * (1.0 - avg_diff)
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Energy & ZCR (auxiliary, not in the default 4-feature weighted score
# but useful in the report for short-time analysis discussion)
# ---------------------------------------------------------------------------

def compare_energy(e1, e2):
    if e1 == 0 and e2 == 0:
        return 100.0
    if e1 == 0 or e2 == 0:
        return 0.0
    diff_ratio = abs(e1 - e2) / max(e1, e2, 1e-10)
    return float(np.clip(100.0 * (1.0 - diff_ratio), 0.0, 100.0))


def compare_zcr(z1, z2):
    if z1 == 0 and z2 == 0:
        return 100.0
    if z1 == 0 or z2 == 0:
        return 0.0
    diff_ratio = abs(z1 - z2) / max(z1, z2, 1e-10)
    return float(np.clip(100.0 * (1.0 - diff_ratio), 0.0, 100.0))