"""
experiments.py
==============
Five comparison types, covering all combinations requested:

  Exp 1 - Same speaker,     same word   (file vs itself)   -> sanity check / ceiling
  Exp 2 - Different speakers, same word                    -> voice-independent check
  Exp 3 - Same speaker,     different words                -> content-discrimination check
  Exp 4 - Different speakers, different words               -> lower bound / floor check
  Exp 5 - Correct vs intentionally incorrect pronunciation -> error-sensitivity check

NOTE ON EXPECTED RANGES: the numbers below are *expectations to sanity-check
against*, not targets the algorithm is tuned to hit. The comparison/scoring
functions in comparison.py and scoring.py compute distances and similarities
directly from the audio with no manually-chosen constants designed to force
a particular output range. Report the numbers you actually measure -- if they
land outside these rough bands, that is useful, honestly-reported information
for your report's Discussion/Limitations section, not a bug to hide.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from audio_utils import preprocess_audio, save_audio
from feature_extraction import extract_all_features
from comparison import compare_mfcc, compare_pitch, compare_duration, compare_formants
from scoring import calculate_score


class Experiments:

    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {}
        self._feature_cache = {}   # path -> features dict

    # ------------------------------------------------------------------
    def _get_features(self, path):
        """Cache features per file path to avoid recomputation across experiments."""
        if path not in self._feature_cache:
            y, sr = preprocess_audio(path)
            if y is None:
                self._feature_cache[path] = None
            else:
                self._feature_cache[path] = extract_all_features(path, y, sr)
        return self._feature_cache[path]

    def _score_pair(self, f1, f2):
        mfcc_s = compare_mfcc(f1['mfcc'], f2['mfcc'])
        pitch_s = compare_pitch(f1['pitch'], f2['pitch'])
        dur_s = compare_duration(f1['duration'], f2['duration'])
        formant_s = compare_formants(f1['formants'], f2['formants'])
        final_s = calculate_score(mfcc_s, pitch_s, dur_s, formant_s)
        return mfcc_s, pitch_s, dur_s, formant_s, final_s

    # ------------------------------------------------------------------
    # Experiment 1 - Same speaker, same word (self-comparison, sanity ceiling)
    # ------------------------------------------------------------------

    def experiment_1_same_speaker_same_word(self):
        print("\n" + "=" * 60)
        print("Experiment 1: Same Speaker, Same Word (self-comparison)")
        print("Sanity check - should approach 100% (max similarity ceiling)")
        print("=" * 60)

        results = []
        for speaker in self.dataset.speakers:
            for word, path in self.dataset.get_speaker_files(speaker).items():
                f = self._get_features(path)
                if f is None:
                    continue
                mfcc_s, pitch_s, dur_s, formant_s, final_s = self._score_pair(f, f)
                results.append({
                    'speaker': speaker, 'word': word,
                    'mfcc_score': mfcc_s, 'pitch_score': pitch_s,
                    'duration_score': dur_s, 'formant_score': formant_s,
                    'final_score': final_s,
                })

        self.results['same_speaker_same_word'] = results
        self._print_results(results, "Same Speaker, Same Word")
        return results

    # ------------------------------------------------------------------
    # Experiment 2 - Different speakers, same word
    # ------------------------------------------------------------------

    def experiment_2_diff_speaker_same_word(self):
        print("\n" + "=" * 60)
        print("Experiment 2: Different Speakers, Same Word")
        print("=" * 60)

        results = []
        for word in self.dataset.get_test_words():
            word_files = self.dataset.get_word_files(word)
            speakers = list(word_files.keys())

            for i in range(len(speakers)):
                for j in range(i + 1, len(speakers)):
                    sp1, sp2 = speakers[i], speakers[j]
                    f1 = self._get_features(word_files[sp1])
                    f2 = self._get_features(word_files[sp2])
                    if f1 is None or f2 is None:
                        continue

                    mfcc_s, pitch_s, dur_s, formant_s, final_s = self._score_pair(f1, f2)
                    results.append({
                        'word': word, 'speaker1': sp1, 'speaker2': sp2,
                        'mfcc_score': mfcc_s, 'pitch_score': pitch_s,
                        'duration_score': dur_s, 'formant_score': formant_s,
                        'final_score': final_s,
                    })

        self.results['diff_speaker_same_word'] = results
        self._print_results(results, "Different Speakers, Same Word")
        return results

    # ------------------------------------------------------------------
    # Experiment 3 - Same speaker, different words
    # ------------------------------------------------------------------

    def experiment_3_same_speaker_diff_word(self):
        print("\n" + "=" * 60)
        print("Experiment 3: Same Speaker, Different Words")
        print("=" * 60)

        results = []
        for speaker in self.dataset.speakers:
            files = self.dataset.get_speaker_files(speaker)
            words = list(files.keys())

            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    f1 = self._get_features(files[w1])
                    f2 = self._get_features(files[w2])
                    if f1 is None or f2 is None:
                        continue

                    mfcc_s, pitch_s, dur_s, formant_s, final_s = self._score_pair(f1, f2)
                    results.append({
                        'speaker': speaker, 'word1': w1, 'word2': w2,
                        'mfcc_score': mfcc_s, 'pitch_score': pitch_s,
                        'duration_score': dur_s, 'formant_score': formant_s,
                        'final_score': final_s,
                    })

        self.results['same_speaker_diff_word'] = results
        self._print_results(results, "Same Speaker, Different Words")
        return results

    # ------------------------------------------------------------------
    # Experiment 4 - Different speakers, different words (lower bound)
    # ------------------------------------------------------------------

    def experiment_4_diff_speaker_diff_word(self, max_pairs_per_speaker_pair=None):
        """
        max_pairs_per_speaker_pair: optional cap to limit combinatorial blowup.
        With 5 speakers x 10 words there are C(5,2)=10 speaker pairs and
        C(10,2)=45 word pairs per speaker pair = 450 comparisons total, which
        is fine to run in full; the cap is provided for larger datasets.
        """
        print("\n" + "=" * 60)
        print("Experiment 4: Different Speakers, Different Words")
        print("=" * 60)

        results = []
        speakers = self.dataset.speakers
        words = self.dataset.get_test_words()

        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                sp1, sp2 = speakers[i], speakers[j]
                files1 = self.dataset.get_speaker_files(sp1)
                files2 = self.dataset.get_speaker_files(sp2)

                pair_count = 0
                for wi in range(len(words)):
                    for wj in range(len(words)):
                        if words[wi] == words[wj]:
                            continue  # different words only
                        if wi >= wj:
                            continue  # avoid duplicate unordered pairs
                        w1, w2 = words[wi], words[wj]
                        if w1 not in files1 or w2 not in files2:
                            continue

                        f1 = self._get_features(files1[w1])
                        f2 = self._get_features(files2[w2])
                        if f1 is None or f2 is None:
                            continue

                        mfcc_s, pitch_s, dur_s, formant_s, final_s = self._score_pair(f1, f2)
                        results.append({
                            'speaker1': sp1, 'word1': w1,
                            'speaker2': sp2, 'word2': w2,
                            'mfcc_score': mfcc_s, 'pitch_score': pitch_s,
                            'duration_score': dur_s, 'formant_score': formant_s,
                            'final_score': final_s,
                        })
                        pair_count += 1
                        if max_pairs_per_speaker_pair and pair_count >= max_pairs_per_speaker_pair:
                            break
                    if max_pairs_per_speaker_pair and pair_count >= max_pairs_per_speaker_pair:
                        break

        self.results['diff_speaker_diff_word'] = results
        self._print_results(results, "Different Speakers, Different Words")
        return results

    # ------------------------------------------------------------------
    # Experiment 5 - Correct vs intentionally incorrect pronunciation
    # ------------------------------------------------------------------

    def experiment_5_correct_vs_incorrect(self):
        """
        Build modified ("incorrect") versions of original recordings and
        compare them against the original (correct) version.

        FIXED: error parameters are now realistic articulation-error
        magnitudes (e.g. +/-2 semitones pitch, 1.15x-1.3x speaking rate),
        not exaggerated "more aggressive" values whose only purpose was to
        force the score into a pre-decided range. Real mispronunciation by
        a learner is rarely a 65% slowdown or a 7-semitone pitch shift; this
        version aims to model plausible learner errors instead.
        """
        print("\n" + "=" * 60)
        print("Experiment 5: Correct vs Incorrect Pronunciation")
        print("=" * 60)

        results = []
        speakers = self.dataset.speakers[:3]
        temp_dir = tempfile.mkdtemp()

        try:
            for speaker in speakers:
                sp_files = self.dataset.get_speaker_files(speaker)
                words = list(sp_files.keys())[:4]

                for word in words:
                    path = sp_files[word]
                    y, sr = preprocess_audio(path)
                    if y is None:
                        continue

                    f_correct = self._get_features(path)
                    errors = self._build_error_versions(y, sr, speaker, word, temp_dir)

                    for error_name, (y_err, path_err) in errors.items():
                        f_err = extract_all_features(path_err, y_err, sr)

                        mfcc_s = compare_mfcc(f_correct['mfcc'], f_err['mfcc'])
                        pitch_s = compare_pitch(f_correct['pitch'], f_err['pitch'])
                        dur_s = compare_duration(f_correct['duration'], f_err['duration'])
                        formant_s = compare_formants(f_correct['formants'], f_err['formants'])
                        final_s = calculate_score(mfcc_s, pitch_s, dur_s, formant_s)

                        results.append({
                            'speaker': speaker, 'word': word,
                            'error_type': error_name,
                            'mfcc_score': mfcc_s, 'pitch_score': pitch_s,
                            'duration_score': dur_s, 'formant_score': formant_s,
                            'final_score': final_s,
                        })
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        self.results['correct_vs_incorrect'] = results
        self._print_results(results, "Correct vs Incorrect")
        self._print_error_breakdown(results)
        return results

    def _build_error_versions(self, y, sr, speaker, word, temp_dir):
        """Plausible learner-level pronunciation errors (realistic magnitudes)."""
        errors = {}

        # 1. Pitch error - +/- 2 semitones (within natural intonation-error range)
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        p1 = os.path.join(temp_dir, f"pitch_{speaker}_{word}.wav")
        save_audio(y_pitch, sr, p1)
        errors['Pitch Error'] = (y_pitch, p1)

        # 2. Duration error - speaking 30% slower (rate=0.77), a realistic
        #    "non-native, drawn-out" pronunciation, not an extreme slowdown
        y_slow = librosa.effects.time_stretch(y, rate=0.77)
        p2 = os.path.join(temp_dir, f"duration_{speaker}_{word}.wav")
        save_audio(y_slow, sr, p2)
        errors['Duration Error'] = (y_slow, p2)

        # 3. Formant error - moderate formant shift (resample-based)
        y_formant = self._formant_shift(y, sr, shift=1.15)
        p3 = os.path.join(temp_dir, f"formant_{speaker}_{word}.wav")
        save_audio(y_formant, sr, p3)
        errors['Formant Error'] = (y_formant, p3)

        # 4. Combined errors - moderate pitch + moderate slowdown + moderate formant shift
        y_comb = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        y_comb = librosa.effects.time_stretch(y_comb, rate=0.85)
        y_comb = self._formant_shift(y_comb, sr, shift=1.1)
        p4 = os.path.join(temp_dir, f"combined_{speaker}_{word}.wav")
        save_audio(y_comb, sr, p4)
        errors['Combined Errors'] = (y_comb, p4)

        return errors

    @staticmethod
    def _formant_shift(y, sr, shift=1.15):
        """Approximate formant shift via resampling (changes resonances)."""
        try:
            new_sr = int(sr * shift)
            y_up = librosa.resample(y, orig_sr=sr, target_sr=new_sr)
            y_shifted = librosa.resample(y_up, orig_sr=new_sr, target_sr=sr)
            if len(y_shifted) > len(y):
                y_shifted = y_shifted[:len(y)]
            else:
                y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)))
            mx = np.max(np.abs(y_shifted))
            y_shifted = y_shifted / (mx + 1e-10) if mx > 0 else y_shifted
            return y_shifted
        except Exception as e:
            print(f"Formant shift failed: {e}")
            return y

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def _print_results(self, results, label):
        if not results:
            print(f"No results for {label}.")
            return

        df = pd.DataFrame(results)
        print(f"\n{label} - {len(results)} comparisons")
        print("-" * 50)
        for col in ('mfcc_score', 'pitch_score', 'duration_score',
                    'formant_score', 'final_score'):
            m, s = df[col].mean(), df[col].std()
            print(f"  {col:<18}: mean={m:.1f}%  std={s:.1f}%")

        total = len(df)
        ex = len(df[df['final_score'] >= 85])
        gd = len(df[(df['final_score'] >= 70) & (df['final_score'] < 85)])
        fa = len(df[(df['final_score'] >= 55) & (df['final_score'] < 70)])
        po = len(df[df['final_score'] < 55])
        print(f"\n  Score distribution:")
        print(f"    Excellent >=85%  : {ex:3d}  ({ex/total*100:.1f}%)")
        print(f"    Good     70-84%  : {gd:3d}  ({gd/total*100:.1f}%)")
        print(f"    Fair     55-69%  : {fa:3d}  ({fa/total*100:.1f}%)")
        print(f"    Poor     <55%    : {po:3d}  ({po/total*100:.1f}%)")

    def _print_error_breakdown(self, results):
        if not results:
            return
        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("ERROR TYPE BREAKDOWN")
        print("=" * 60)

        avg_scores = {}
        for etype in df['error_type'].unique():
            sub = df[df['error_type'] == etype]
            avg = sub['final_score'].mean()
            avg_scores[etype] = avg
            print(f"\n{etype}  (n={len(sub)})")
            for col in ('mfcc_score', 'pitch_score', 'duration_score',
                        'formant_score', 'final_score'):
                print(f"  {col:<18}: {sub[col].mean():.1f}%")

        print("\nRanked by impact (lowest score = most impactful error):")
        for i, (et, sc) in enumerate(sorted(avg_scores.items(), key=lambda x: x[1]), 1):
            print(f"  {i}. {et}: {sc:.1f}%")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Arabic Pronunciation Assessment - All Comparison Types",
                     fontsize=14, fontweight='bold')

        ax1 = axes[0, 0]
        labels = ['Same Sp.\nSame Wd', 'Diff Sp.\nSame Wd',
                  'Same Sp.\nDiff Wd', 'Diff Sp.\nDiff Wd', 'Correct vs\nIncorrect']
        keys = ['same_speaker_same_word', 'diff_speaker_same_word',
                'same_speaker_diff_word', 'diff_speaker_diff_word',
                'correct_vs_incorrect']
        means, stds = [], []
        for k in keys:
            if k in self.results and self.results[k]:
                s = [r['final_score'] for r in self.results[k]]
                means.append(np.mean(s)); stds.append(np.std(s))
            else:
                means.append(0); stds.append(0)

        colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel('Average Final Score (%)')
        ax1.set_title('Average Score by Comparison Type')
        ax1.tick_params(axis='x', labelsize=8)
        for bar, m in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                      f'{m:.1f}%', ha='center', va='bottom', fontsize=8)

        ax2 = axes[0, 1]
        if self.results.get('correct_vs_incorrect'):
            df = pd.DataFrame(self.results['correct_vs_incorrect'])
            etypes = df['error_type'].unique()
            e_means = [df[df['error_type'] == e]['final_score'].mean() for e in etypes]
            e_stds = [df[df['error_type'] == e]['final_score'].std() for e in etypes]
            ax2.bar(etypes, e_means, yerr=e_stds, capsize=5,
                    color=['#3498db', '#e67e22', '#2ecc71', '#e74c3c'])
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Average Final Score (%)')
            ax2.set_title('Score by Error Type')
            ax2.tick_params(axis='x', rotation=15, labelsize=8)
        else:
            ax2.text(0.5, 0.5, 'Exp 5 not run yet', ha='center', va='center',
                      transform=ax2.transAxes)

        ax3 = axes[0, 2]
        if self.results.get('correct_vs_incorrect'):
            comps = ['mfcc_score', 'pitch_score', 'duration_score', 'formant_score']
            clabels = ['MFCC', 'Pitch', 'Duration', 'Formants']
            df = pd.DataFrame(self.results['correct_vs_incorrect'])
            c_means = [df[c].mean() for c in comps]
            c_stds = [df[c].std() for c in comps]
            ax3.bar(clabels, c_means, yerr=c_stds, capsize=5,
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('Average Score (%)')
            ax3.set_title('Component Scores (Exp 5)')
        else:
            ax3.text(0.5, 0.5, 'Exp 5 not run yet', ha='center', va='center',
                      transform=ax3.transAxes)

        ax4 = axes[1, 0]
        if self.results.get('same_speaker_same_word'):
            sp_scores = {}
            for r in self.results['same_speaker_same_word']:
                sp_scores.setdefault(r['speaker'], []).append(r['final_score'])
            sps = list(sp_scores.keys())
            s_m = [np.mean(sp_scores[s]) for s in sps]
            s_std = [np.std(sp_scores[s]) for s in sps]
            ax4.bar(sps, s_m, yerr=s_std, capsize=5, color='#2ecc71')
            ax4.set_ylim(0, 105)
            ax4.set_xlabel('Speaker')
            ax4.set_ylabel('Average Score (%)')
            ax4.set_title('Exp 1 - Self-Comparison by Speaker')
            ax4.tick_params(axis='x', rotation=30, labelsize=8)
        else:
            ax4.text(0.5, 0.5, 'Exp 1 not run yet', ha='center', va='center',
                      transform=ax4.transAxes)

        ax5 = axes[1, 1]
        if self.results.get('same_speaker_diff_word'):
            df = pd.DataFrame(self.results['same_speaker_diff_word'])
            word_scores = df.groupby('word1')['final_score'].mean() if 'word1' in df else None
            df.boxplot(column='final_score', by='speaker', ax=ax5, rot=30)
            ax5.set_title('Exp 3 - Same Speaker, Diff Words (by speaker)')
            ax5.set_xlabel('Speaker')
            ax5.set_ylabel('Final Score (%)')
            plt.sca(ax5)
        else:
            ax5.text(0.5, 0.5, 'Exp 3 not run yet', ha='center', va='center',
                      transform=ax5.transAxes)

        ax6 = axes[1, 2]
        if self.results.get('diff_speaker_diff_word'):
            s = [r['final_score'] for r in self.results['diff_speaker_diff_word']]
            ax6.hist(s, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Final Score (%)')
            ax6.set_ylabel('Count')
            ax6.set_title('Exp 4 - Diff Speaker/Diff Word Distribution')
        else:
            ax6.text(0.5, 0.5, 'Exp 4 not run yet', ha='center', va='center',
                      transform=ax6.transAxes)

        plt.tight_layout()
        plt.savefig('results/all_experiments.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_similarity_matrix(self):
        if not self.results.get('diff_speaker_same_word'):
            print("Run Experiment 2 first.")
            return

        speakers = self.dataset.speakers
        n = len(speakers)
        mat = np.zeros((n, n))
        count = np.zeros((n, n))

        for r in self.results['diff_speaker_same_word']:
            i = speakers.index(r['speaker1']) if r['speaker1'] in speakers else -1
            j = speakers.index(r['speaker2']) if r['speaker2'] in speakers else -1
            if i >= 0 and j >= 0:
                mat[i, j] += r['final_score']
                mat[j, i] += r['final_score']
                count[i, j] += 1
                count[j, i] += 1

        with np.errstate(invalid='ignore'):
            mat = np.where(count > 0, mat / count, 0)
        np.fill_diagonal(mat, 100)

        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt='.1f',
                    xticklabels=speakers, yticklabels=speakers,
                    cmap='RdYlGn', vmin=0, vmax=100)
        plt.title('Speaker Similarity Matrix (Same-Word Comparisons)')
        plt.tight_layout()
        plt.savefig('results/similarity_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all_experiments(self):
        print("\n" + "=" * 60)
        print("RUNNING ALL EXPERIMENTS")
        print("=" * 60)
        self.experiment_1_same_speaker_same_word()
        self.experiment_2_diff_speaker_same_word()
        self.experiment_3_same_speaker_diff_word()
        self.experiment_4_diff_speaker_diff_word()
        self.experiment_5_correct_vs_incorrect()
        self.plot_results()
        self.plot_similarity_matrix()
        self._print_summary()

    def _print_summary(self):
        """
        Print a comprehensive summary comparing measured results against
        rough sanity-check expectations (see module docstring).
        """
        print("\n" + "=" * 60)
        print("SUMMARY - measured results vs. sanity-check expectations")
        print("=" * 60)

        expected = {
            'same_speaker_same_word':  (95, 100),
            'diff_speaker_same_word':  (55, 80),
            'same_speaker_diff_word':  (15, 45),
            'diff_speaker_diff_word':  (5,  35),
            'correct_vs_incorrect':    (40, 70),
        }

        for key, (lo, hi) in expected.items():
            if key in self.results and self.results[key]:
                scores = [r['final_score'] for r in self.results[key]]
                mean = np.mean(scores)
                label = key.replace('_', ' ').title()
                status = "within rough expected band" if lo <= mean <= hi else \
                         "outside rough expected band (report & discuss this)"
                print(f"\n{label}:")
                print(f"  Rough expectation: {lo}-{hi}%")
                print(f"  Measured mean:     {mean:.1f}%   -> {status}")