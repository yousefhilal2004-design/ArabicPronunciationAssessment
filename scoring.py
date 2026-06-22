"""
scoring.py
==========
Weights per project spec:
  MFCC      40 %  – spectral shape
  Pitch     15 %  – prosody / intonation
  Duration  15 %  – temporal structure
  Formants  20 %  – vowel quality
  
Total = 90 % (the weights sum to 90%, giving max score of 90%)
"""

import numpy as np


DEFAULT_WEIGHTS = {
    'mfcc':     0.40,
    'pitch':    0.15,
    'duration': 0.15,
    'formants': 0.20,
}


def calculate_score(mfcc_score, pitch_score, duration_score, formant_score,
                    weights=None, verbose=False):
    """
    Weighted average pronunciation score (0–100).
    
    The weights sum to 90% (0.40 + 0.15 + 0.15 + 0.20 = 0.90).
    Maximum possible score is 90%.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # Clamp components to valid range
    scores = {
        'mfcc':     float(np.clip(mfcc_score,     0, 100)),
        'pitch':    float(np.clip(pitch_score,    0, 100)),
        'duration': float(np.clip(duration_score, 0, 100)),
        'formants': float(np.clip(formant_score,  0, 100)),
    }
    
    # Calculate weighted sum - NO NORMALIZATION
    final = (
        weights['mfcc']     * scores['mfcc']     +
        weights['pitch']    * scores['pitch']    +
        weights['duration'] * scores['duration'] +
        weights['formants'] * scores['formants']
    )
    
    # Round to 1 decimal place
    final = round(float(np.clip(final, 0, 100)), 1)
    
    if verbose:
        print("\nScore Breakdown:")
        for k in ('mfcc', 'pitch', 'duration', 'formants'):
            weight = weights.get(k, 0)
            contribution = weight * scores[k]
            print(f"  {k.capitalize():<10}: "
                  f"{scores[k]:6.1f}% × {weight:.2f} = {contribution:5.1f}")
        print(f"  {'Final':<10}: {final:.1f}% (max possible: 90.0%)")
    
    return final


def generate_feedback(score, mfcc_score=None, pitch_score=None,
                      duration_score=None, formant_score=None):
    """Generate human-readable feedback."""
    lines = []
    
    # Adjusted thresholds for 90% max score
    if score >= 85:  # 85/90 = 94%
        lines.append("🌟 EXCELLENT — pronunciation closely matches the reference.")
    elif score >= 70:  # 70/90 = 78%
        lines.append("👍 GOOD — minor differences that don't affect clarity.")
    elif score >= 55:  # 55/90 = 61%
        lines.append("📖 FAIR — understandable but improvement needed.")
    else:
        lines.append("⚠️ NEEDS IMPROVEMENT — pronunciation differs significantly.")
    
    if mfcc_score is not None and mfcc_score < 55:
        lines.append("🎵 Spectral shape (MFCC) differs significantly — check overall articulation.")
    elif mfcc_score is not None and mfcc_score < 72:
        lines.append("🎵 Moderate spectral differences detected.")
    
    if pitch_score is not None and pitch_score < 55:
        lines.append("🎤 Pitch contour differs significantly — check intonation and stress.")
    elif pitch_score is not None and pitch_score < 72:
        lines.append("🎤 Some pitch variation compared to reference.")
    
    if duration_score is not None and duration_score < 55:
        lines.append("⏱️ Duration differs significantly — check vowel length and speaking rate.")
    elif duration_score is not None and duration_score < 72:
        lines.append("⏱️ Minor timing differences detected.")
    
    if formant_score is not None and formant_score < 55:
        lines.append("🔊 Formant differences detected — check vowel quality and tongue position.")
    elif formant_score is not None and formant_score < 72:
        lines.append("🔊 Some vowel quality differences.")
    
    if score < 70:
        lines.append("💡 Practice the target sounds and compare with native pronunciation.")
    
    return "\n".join(lines)


def get_calibrated_weights():
    """Return weight sets for different Arabic sound categories."""
    return {
        'standard': {
            'mfcc': 0.40, 'pitch': 0.15,
            'duration': 0.15, 'formants': 0.20,
        },
        'emphatic_focus': {
            'mfcc': 0.45, 'pitch': 0.10,
            'duration': 0.15, 'formants': 0.20,
        },
        'pharyngeal_focus': {
            'mfcc': 0.35, 'pitch': 0.20,
            'duration': 0.15, 'formants': 0.20,
        },
        'uvular_focus': {
            'mfcc': 0.35, 'pitch': 0.10,
            'duration': 0.20, 'formants': 0.25,
        },
    }