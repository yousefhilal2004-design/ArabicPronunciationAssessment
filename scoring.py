"""
scoring.py
==========
Weighted final pronunciation score, per project spec example:
  "40% MFCC similarity, 20% Pitch similarity,
   20% Duration similarity, 20% Formant similarity"

FIXED: weights now sum to exactly 1.00 (100%).
The original file used 0.40+0.15+0.15+0.20 = 0.90, capping the maximum
possible score at 90% even for a perfect match against itself — that is
a bug, not a calibration choice, and has been corrected here to match
the example weighting given in the project description.
"""

import numpy as np


DEFAULT_WEIGHTS = {
    'mfcc':     0.40,
    'pitch':    0.20,
    'duration': 0.20,
    'formants': 0.20,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 100%"


def calculate_score(mfcc_score, pitch_score, duration_score, formant_score,
                     weights=None, verbose=False):
    """
    Weighted average pronunciation score (0-100).
    Weights sum to 1.00 -> max possible score is 100%.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    scores = {
        'mfcc':     float(np.clip(mfcc_score,     0, 100)),
        'pitch':    float(np.clip(pitch_score,    0, 100)),
        'duration': float(np.clip(duration_score, 0, 100)),
        'formants': float(np.clip(formant_score,  0, 100)),
    }

    final = (
        weights['mfcc']     * scores['mfcc']     +
        weights['pitch']    * scores['pitch']    +
        weights['duration'] * scores['duration'] +
        weights['formants'] * scores['formants']
    )

    final = round(float(np.clip(final, 0, 100)), 1)

    if verbose:
        print("\nScore Breakdown:")
        for k in ('mfcc', 'pitch', 'duration', 'formants'):
            weight = weights.get(k, 0)
            contribution = weight * scores[k]
            print(f"  {k.capitalize():<10}: "
                  f"{scores[k]:6.1f}% x {weight:.2f} = {contribution:5.1f}")
        print(f"  {'Final':<10}: {final:.1f}%")

    return final


def generate_feedback(score, mfcc_score=None, pitch_score=None,
                       duration_score=None, formant_score=None):
    """Generate human-readable feedback based on a 0-100 scale."""
    lines = []

    if score >= 85:
        lines.append("EXCELLENT - pronunciation closely matches the reference.")
    elif score >= 70:
        lines.append("GOOD - minor differences that don't affect clarity.")
    elif score >= 55:
        lines.append("FAIR - understandable but improvement needed.")
    else:
        lines.append("NEEDS IMPROVEMENT - pronunciation differs significantly.")

    if mfcc_score is not None and mfcc_score < 55:
        lines.append("Spectral shape (MFCC) differs significantly - check overall articulation.")
    elif mfcc_score is not None and mfcc_score < 72:
        lines.append("Moderate spectral differences detected.")

    if pitch_score is not None and pitch_score < 55:
        lines.append("Pitch contour differs significantly - check intonation and stress.")
    elif pitch_score is not None and pitch_score < 72:
        lines.append("Some pitch variation compared to reference.")

    if duration_score is not None and duration_score < 55:
        lines.append("Duration differs significantly - check vowel length and speaking rate.")
    elif duration_score is not None and duration_score < 72:
        lines.append("Minor timing differences detected.")

    if formant_score is not None and formant_score < 55:
        lines.append("Formant differences detected - check vowel quality and tongue position.")
    elif formant_score is not None and formant_score < 72:
        lines.append("Some vowel quality differences.")

    if score < 70:
        lines.append("Practice the target sounds and compare with native pronunciation.")

    return "\n".join(lines)


def get_calibrated_weights():
    """
    Alternative weight sets for different Arabic sound categories.
    All sets sum to 1.00.
    """
    return {
        'standard': {
            'mfcc': 0.40, 'pitch': 0.20,
            'duration': 0.20, 'formants': 0.20,
        },
        'emphatic_focus': {       # ط ض ص ظ ق -> spectral shape matters most
            'mfcc': 0.50, 'pitch': 0.10,
            'duration': 0.15, 'formants': 0.25,
        },
        'pharyngeal_focus': {     # ح ع -> pitch/voicing quality matters more
            'mfcc': 0.35, 'pitch': 0.25,
            'duration': 0.15, 'formants': 0.25,
        },
        'uvular_focus': {         # خ غ -> formant/place of articulation
            'mfcc': 0.35, 'pitch': 0.15,
            'duration': 0.20, 'formants': 0.30,
        },
    }