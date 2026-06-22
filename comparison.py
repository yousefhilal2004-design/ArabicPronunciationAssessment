"""
comparison.py
=============
All comparison functions return a score in [0, 100].

Target score ranges:
  Experiment 1 – Same speaker vs same speaker  → 85–100 %
  Experiment 2 – Different speakers, same word  → 65–80 %
  Experiment 3 – Correct vs incorrect          → 35–60 %
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


# ---------------------------------------------------------------------------
# DTW Implementation (manual, no external dependency)
# ---------------------------------------------------------------------------

def dtw_distance(seq1, seq2):
    """
    Compute DTW distance between two sequences using dynamic programming.
    Handles sequences of different lengths.
    """
    n = len(seq1)
    m = len(seq2)
    
    # If sequences are empty, return 0
    if n == 0 or m == 0:
        return 0
    
    # Initialize DP matrix with infinity
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between points
            cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
    
    return dtw[n, m]


def compare_mfcc(mfcc1, mfcc2, method='dtw'):
    """
    Compare two MFCC matrices.
    
    Method 'dtw':
      - Use Dynamic Time Warping to compare full sequences
      - Distance is normalized by the number of frames
      - Identical signals → ~100%
      - Different speakers → ~65-75%
      - Very different → ~40-50%
    """
    if mfcc1 is None or mfcc2 is None:
        return 50.0
    if mfcc1.shape[1] == 0 or mfcc2.shape[1] == 0:
        return 50.0
    
    if method == 'dtw':
        try:
            # Transpose so each column is a time frame
            seq1 = mfcc1.T  # shape: (frames, n_mfcc)
            seq2 = mfcc2.T  # shape: (frames, n_mfcc)
            
            # Normalize MFCC sequences to have similar scale
            seq1 = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-10)
            seq2 = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-10)
            
            # Compute DTW distance
            distance = dtw_distance(seq1, seq2)
            
            # Normalize by the number of frames
            n1 = mfcc1.shape[1]
            n2 = mfcc2.shape[1]
            max_frames = max(n1, n2, 1)
            
            # Scale factor controls sensitivity
            # Lower = more strict (better for error detection)
            scale_factor = 0.8
            
            # Normalize distance
            normalized_distance = distance / (max_frames * scale_factor + 1e-10)
            
            # Convert to similarity (exponential decay)
            # More aggressive decay for better error detection
            similarity = 100.0 * np.exp(-normalized_distance * 0.6)
            
            # Clip to valid range
            similarity = max(0, min(100, similarity))
            
            return float(similarity)
            
        except Exception as e:
            print(f"DTW error, using fallback: {e}")
            return compare_mfcc_cosine(mfcc1, mfcc2)
    
    else:
        return compare_mfcc_cosine(mfcc1, mfcc2)


def compare_mfcc_cosine(mfcc1, mfcc2):
    """Fallback: Compare MFCC means using cosine similarity."""
    mean1 = np.mean(mfcc1, axis=1).reshape(1, -1)
    mean2 = np.mean(mfcc2, axis=1).reshape(1, -1)
    sim = float(cosine_similarity(mean1, mean2)[0][0])
    sim = np.clip(sim, -1.0, 1.0)
    score = (sim + 1.0) / 2.0 * 100.0
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Pitch (scalar)
# ---------------------------------------------------------------------------

def compare_pitch(p1, p2):
    """
    Compare two mean pitch values (Hz).
    """
    if p1 == 0 and p2 == 0:
        return 100.0
    if p1 == 0 or p2 == 0:
        return 55.0
    
    diff_ratio = abs(p1 - p2) / max(p1, p2, 1.0)
    score = 100.0 - diff_ratio * 60.0
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Pitch contour (array)
# ---------------------------------------------------------------------------

def compare_pitch_contour(pitch1, pitch2):
    """
    Compare full pitch contours using Pearson correlation.
    """
    if len(pitch1) == 0 or len(pitch2) == 0:
        return 60.0
    
    p1 = pitch1[pitch1 > 0]
    p2 = pitch2[pitch2 > 0]
    
    if len(p1) < 3 or len(p2) < 3:
        return 60.0
    
    min_len = min(len(p1), len(p2))
    p1 = p1[:min_len]
    p2 = p2[:min_len]
    
    try:
        corr = np.corrcoef(p1, p2)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        score = (corr + 1.0) / 2.0 * 100.0
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return 50.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def compare_duration(d1, d2):
    """
    Compare utterance durations.
    """
    if d1 == 0 and d2 == 0:
        return 100.0
    if d1 == 0 or d2 == 0:
        return 60.0
    
    diff_ratio = abs(d1 - d2) / max(d1, d2, 1.0)
    score = 100.0 - diff_ratio * 50.0
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Formants
# ---------------------------------------------------------------------------

def compare_formants(f1, f2):
    """
    Compare F1, F2, F3 triplets.
    """
    f1 = np.array(f1, dtype=float)
    f2 = np.array(f2, dtype=float)
    
    # Replace zeros with neutral value
    f1 = np.where(f1 == 0, 500.0, f1)
    f2 = np.where(f2 == 0, 500.0, f2)
    
    n = min(len(f1), len(f2))
    if n == 0:
        return 50.0
    
    diffs = []
    for i in range(n):
        max_f = max(f1[i], f2[i], 1.0)
        diffs.append(abs(f1[i] - f2[i]) / max_f)
    
    avg_diff = float(np.mean(diffs))
    score = 100.0 - avg_diff * 80.0
    return float(np.clip(score, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Energy & ZCR (auxiliary)
# ---------------------------------------------------------------------------

def compare_energy(e1, e2):
    """Compare total energy values."""
    if e1 == 0 and e2 == 0:
        return 100.0
    if e1 == 0 or e2 == 0:
        return 60.0
    diff_ratio = abs(e1 - e2) / max(e1, e2, 1e-10)
    score = 100.0 - diff_ratio * 40.0
    return float(np.clip(score, 0.0, 100.0))


def compare_zcr(z1, z2):
    """Compare zero-crossing rates."""
    if z1 == 0 and z2 == 0:
        return 100.0
    if z1 == 0 or z2 == 0:
        return 60.0
    diff_ratio = abs(z1 - z2) / max(z1, z2, 1e-10)
    score = 100.0 - diff_ratio * 40.0
    return float(np.clip(score, 0.0, 100.0))