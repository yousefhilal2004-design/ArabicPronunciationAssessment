import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import parselmouth
import sys
from io import StringIO

from audio_utils import preprocess_audio, load_audio
from feature_extraction import (
    extract_mfcc, extract_pitch, extract_duration, 
    extract_formants, extract_energy, extract_zcr,
    extract_pitch_contour, extract_all_features
)
from comparison import (
    compare_mfcc, compare_pitch, compare_duration, 
    compare_formants, compare_pitch_contour, compare_energy, compare_zcr
)
from scoring import calculate_score, generate_feedback
from dataset import ArabicDataset
from experiments import Experiments


class PronunciationGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Pronunciation Assessment System")
        self.root.geometry("1200x900")
        
        self.audio1 = ""
        self.audio2 = ""
        self.dataset = None
        self.experiments = None
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        self.main_tab = tk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="🎤 Pronunciation Assessment")
        
        # Dataset tab
        self.dataset_tab = tk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="📁 Dataset Management")
        
        # Experiments tab
        self.experiments_tab = tk.Frame(self.notebook)
        self.notebook.add(self.experiments_tab, text="🧪 Experiments")
        
        # Results tab
        self.results_tab = tk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📊 Results")
        
        self._setup_main_tab()
        self._setup_dataset_tab()
        self._setup_experiments_tab()
        self._setup_results_tab()
    
    def _setup_main_tab(self):
        """Setup the main pronunciation assessment tab"""
        # Title
        title = tk.Label(
            self.main_tab,
            text="🎤 Arabic Pronunciation Assessment System",
            font=("Arial", 22, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=20)
        
        # Subtitle
        subtitle = tk.Label(
            self.main_tab,
            text="Compare reference pronunciation with student pronunciation",
            font=("Arial", 12),
            fg="#7f8c8d"
        )
        subtitle.pack(pady=(0, 20))
        
        # Main frame for audio selection
        audio_frame = tk.Frame(self.main_tab)
        audio_frame.pack(pady=10)
        
        # Left side - Reference Audio
        ref_frame = tk.LabelFrame(
            audio_frame,
            text="📖 Reference Audio",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=15,
            width=400,
            height=200
        )
        ref_frame.pack(side=tk.LEFT, padx=20)
        ref_frame.pack_propagate(False)
        
        btn_audio1 = tk.Button(
            ref_frame,
            text="Select Reference Audio",
            width=25,
            height=2,
            command=self.select_audio1,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold")
        )
        btn_audio1.pack(pady=10)
        
        self.audio1_label = tk.Label(
            ref_frame,
            text="No Reference Audio Selected",
            font=("Arial", 10),
            fg="#7f8c8d",
            wraplength=350
        )
        self.audio1_label.pack(pady=5)
        
        self.audio1_info = tk.Label(
            ref_frame,
            text="",
            font=("Arial", 9),
            fg="#2ecc71"
        )
        self.audio1_info.pack(pady=5)
        
        # Right side - Student Audio
        stud_frame = tk.LabelFrame(
            audio_frame,
            text="🎓 Student Audio",
            font=("Arial", 12, "bold"),
            padx=20,
            pady=15,
            width=400,
            height=200
        )
        stud_frame.pack(side=tk.RIGHT, padx=20)
        stud_frame.pack_propagate(False)
        
        btn_audio2 = tk.Button(
            stud_frame,
            text="Select Student Audio",
            width=25,
            height=2,
            command=self.select_audio2,
            bg="#e67e22",
            fg="white",
            font=("Arial", 10, "bold")
        )
        btn_audio2.pack(pady=10)
        
        self.audio2_label = tk.Label(
            stud_frame,
            text="No Student Audio Selected",
            font=("Arial", 10),
            fg="#7f8c8d",
            wraplength=350
        )
        self.audio2_label.pack(pady=5)
        
        self.audio2_info = tk.Label(
            stud_frame,
            text="",
            font=("Arial", 9),
            fg="#2ecc71"
        )
        self.audio2_info.pack(pady=5)
        
        # Compare button
        btn_compare = tk.Button(
            self.main_tab,
            text="🔍 Compare Pronunciations",
            width=30,
            height=2,
            bg="#27ae60",
            fg="white",
            font=("Arial", 14, "bold"),
            command=self.analyze
        )
        btn_compare.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.LabelFrame(
            self.main_tab,
            text="📊 Assessment Results",
            font=("Arial", 13, "bold"),
            padx=15,
            pady=15
        )
        self.results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Canvas with scrollbar for results
        canvas = tk.Canvas(self.results_frame)
        scrollbar = tk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.result_label = tk.Label(
            self.scrollable_frame,
            text="Select two audio files and click 'Compare Pronunciations' to start assessment.",
            font=("Arial", 12),
            justify="left",
            fg="#7f8c8d"
        )
        self.result_label.pack(pady=20, padx=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _setup_dataset_tab(self):
        """Setup the dataset management tab"""
        title = tk.Label(
            self.dataset_tab,
            text="📁 Dataset Management",
            font=("Arial", 18, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=10)
        
        # Dataset path
        path_frame = tk.Frame(self.dataset_tab)
        path_frame.pack(pady=15)
        
        tk.Label(
            path_frame, 
            text="Dataset Path:",
            font=("Arial", 11, "bold")
        ).pack(side=tk.LEFT, padx=5)
        
        self.path_entry = tk.Entry(path_frame, width=50, font=("Arial", 10))
        self.path_entry.pack(side=tk.LEFT, padx=5)
        self.path_entry.insert(0, "./dataset")
        
        btn_load = tk.Button(
            path_frame,
            text="📂 Load Dataset",
            command=self.load_dataset,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15
        )
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_refresh = tk.Button(
            path_frame,
            text="🔄 Refresh",
            command=self.refresh_dataset,
            bg="#95a5a6",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10
        )
        btn_refresh.pack(side=tk.LEFT, padx=5)
        
        # Dataset info
        self.dataset_info = tk.Label(
            self.dataset_tab,
            text="No dataset loaded. Click 'Load Dataset' to load your dataset.",
            font=("Arial", 11),
            justify="left",
            fg="#7f8c8d"
        )
        self.dataset_info.pack(pady=15, padx=20)
        
        # Audio list frame
        list_frame = tk.LabelFrame(
            self.dataset_tab,
            text="📋 Available Audio Files",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create Treeview for audio list
        columns = ('Speaker', 'Word (Arabic)', 'File Path')
        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.tree.heading('Speaker', text='Speaker')
        self.tree.heading('Word (Arabic)', text='Word (Arabic)')
        self.tree.heading('File Path', text='File Path')
        
        self.tree.column('Speaker', width=120, anchor='center')
        self.tree.column('Word (Arabic)', width=150, anchor='center')
        self.tree.column('File Path', width=400, anchor='w')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Add double-click event to play audio
        self.tree.bind('<Double-Button-1>', self.play_selected_audio)
        
        # Button frame for dataset operations
        btn_frame = tk.Frame(self.dataset_tab)
        btn_frame.pack(pady=10)
        
        btn_play = tk.Button(
            btn_frame,
            text="▶ Play Selected Audio",
            command=self.play_selected_audio,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 10)
        )
        btn_play.pack(side=tk.LEFT, padx=5)
        
        btn_info = tk.Button(
            btn_frame,
            text="ℹ️ Dataset Info",
            command=self.show_dataset_info,
            bg="#3498db",
            fg="white",
            font=("Arial", 10)
        )
        btn_info.pack(side=tk.LEFT, padx=5)
        
        btn_export = tk.Button(
            btn_frame,
            text="📊 Export Dataset Info",
            command=self.export_dataset_info,
            bg="#e67e22",
            fg="white",
            font=("Arial", 10)
        )
        btn_export.pack(side=tk.LEFT, padx=5)
    
    def _setup_experiments_tab(self):
        """Setup the experiments tab"""
        title = tk.Label(
            self.experiments_tab,
            text="🧪 Experiments",
            font=("Arial", 18, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=10)
        
        subtitle = tk.Label(
            self.experiments_tab,
            text="Run experiments to evaluate the pronunciation assessment system",
            font=("Arial", 11),
            fg="#7f8c8d"
        )
        subtitle.pack(pady=(0, 15))
        
        # Experiment buttons frame
        btn_frame = tk.LabelFrame(
            self.experiments_tab,
            text="Experiment Controls",
            font=("Arial", 12, "bold"),
            padx=15,
            pady=15
        )
        btn_frame.pack(pady=10, padx=20, fill='x')
        
        # Experiment 1
        exp1_frame = tk.Frame(btn_frame)
        exp1_frame.pack(fill='x', pady=5)
        
        tk.Label(
            exp1_frame,
            text="1️⃣ Same Speaker vs Same Speaker",
            font=("Arial", 11, "bold"),
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)
        
        tk.Label(
            exp1_frame,
            text="Expected: High similarity scores (≥90%)",
            font=("Arial", 9),
            fg="#27ae60"
        ).pack(side=tk.LEFT, padx=10)
        
        btn_exp1 = tk.Button(
            exp1_frame,
            text="▶ Run",
            command=self.run_experiment_1,
            bg="#2ecc71",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15
        )
        btn_exp1.pack(side=tk.RIGHT)
        
        # Experiment 2
        exp2_frame = tk.Frame(btn_frame)
        exp2_frame.pack(fill='x', pady=5)
        
        tk.Label(
            exp2_frame,
            text="2️⃣ Different Speakers (Same Word)",
            font=("Arial", 11, "bold"),
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)
        
        tk.Label(
            exp2_frame,
            text="Expected: Moderate similarity scores (60-80%)",
            font=("Arial", 9),
            fg="#f39c12"
        ).pack(side=tk.LEFT, padx=10)
        
        btn_exp2 = tk.Button(
            exp2_frame,
            text="▶ Run",
            command=self.run_experiment_2,
            bg="#f39c12",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15
        )
        btn_exp2.pack(side=tk.RIGHT)
        
        # Experiment 3
        exp3_frame = tk.Frame(btn_frame)
        exp3_frame.pack(fill='x', pady=5)
        
        tk.Label(
            exp3_frame,
            text="3️⃣ Correct vs Incorrect Pronunciation",
            font=("Arial", 11, "bold"),
            width=30,
            anchor='w'
        ).pack(side=tk.LEFT)
        
        tk.Label(
            exp3_frame,
            text="Expected: Lower similarity scores (40-60%)",
            font=("Arial", 9),
            fg="#e74c3c"
        ).pack(side=tk.LEFT, padx=10)
        
        btn_exp3 = tk.Button(
            exp3_frame,
            text="▶ Run",
            command=self.run_experiment_3,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15
        )
        btn_exp3.pack(side=tk.RIGHT)
        
        # Run all experiments
        all_frame = tk.Frame(btn_frame)
        all_frame.pack(fill='x', pady=15)
        
        btn_all = tk.Button(
            all_frame,
            text="🚀 RUN ALL EXPERIMENTS",
            command=self.run_all_experiments,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=30,
            height=2
        )
        btn_all.pack()
        
        # Results text area
        results_frame = tk.LabelFrame(
            self.experiments_tab,
            text="Experiment Output",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        results_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.experiment_results = tk.Text(
            text_frame,
            height=15,
            font=("Courier", 10),
            wrap=tk.WORD,
            bg="#f8f9fa"
        )
        
        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.experiment_results.yview)
        self.experiment_results.configure(yscrollcommand=scrollbar.set)
        
        self.experiment_results.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Add initial message
        self.experiment_results.insert(tk.END, "Run an experiment to see results here...\n")
        self.experiment_results.config(state=tk.DISABLED)
    
    def _setup_results_tab(self):
        """Setup the results tab for viewing saved results"""
        title = tk.Label(
            self.results_tab,
            text="📊 Results Archive",
            font=("Arial", 18, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=10)
        
        # Results list
        list_frame = tk.LabelFrame(
            self.results_tab,
            text="Previous Results",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        columns = ('Date', 'Type', 'Score', 'Details')
        self.results_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        self.results_tree.pack(fill='both', expand=True)
        
        # View button
        btn_view = tk.Button(
            self.results_tab,
            text="View Selected Result",
            command=self.view_result,
            bg="#3498db",
            fg="white",
            font=("Arial", 10)
        )
        btn_view.pack(pady=10)
    
    def select_audio1(self):
        self.audio1 = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a"), ("All Files", "*.*")]
        )
        if self.audio1:
            self.audio1_label.config(text=os.path.basename(self.audio1), fg="#2c3e50")
            self.audio1_info.config(text=f"✓ File loaded: {os.path.getsize(self.audio1) // 1024} KB")
    
    def select_audio2(self):
        self.audio2 = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a"), ("All Files", "*.*")]
        )
        if self.audio2:
            self.audio2_label.config(text=os.path.basename(self.audio2), fg="#2c3e50")
            self.audio2_info.config(text=f"✓ File loaded: {os.path.getsize(self.audio2) // 1024} KB")
    
    def analyze(self):
        if self.audio1 == "" or self.audio2 == "":
            messagebox.showerror("Error", "Please select both audio files.")
            return
        
        try:
            # Update status
            self.result_label.config(text="⏳ Processing audio files...", fg="#3498db")
            self.root.update()
            
            # Preprocess audio
            audio1_y, audio1_sr = preprocess_audio(self.audio1)
            audio2_y, audio2_sr = preprocess_audio(self.audio2)
            
            if audio1_y is None or audio2_y is None:
                messagebox.showerror("Error", "Failed to load audio files.")
                return
            
            # Extract features
            features1 = extract_all_features(self.audio1, audio1_y, audio1_sr)
            features2 = extract_all_features(self.audio2, audio2_y, audio2_sr)
            
            # Compare features
            mfcc_score = compare_mfcc(features1['mfcc'], features2['mfcc'])
            pitch_score = compare_pitch(features1['pitch'], features2['pitch'])
            duration_score = compare_duration(features1['duration'], features2['duration'])
            formant_score = compare_formants(features1['formants'], features2['formants'])
            
            # Calculate final score
            final_score = calculate_score(mfcc_score, pitch_score, duration_score, formant_score)
            
            # Generate feedback
            feedback = generate_feedback(
                final_score, 
                mfcc_score=mfcc_score,
                pitch_score=pitch_score,
                duration_score=duration_score,
                formant_score=formant_score
            )
            
            # Determine grade
            if final_score >= 90:
                grade = "🌟 Excellent"
                grade_color = "#27ae60"
            elif final_score >= 75:
                grade = "👍 Good"
                grade_color = "#2ecc71"
            elif final_score >= 60:
                grade = "📖 Fair"
                grade_color = "#f39c12"
            else:
                grade = "⚠️ Needs Improvement"
                grade_color = "#e74c3c"
            
            # Display results
            result_text = (
                f"{'='*60}\n"
                f"🎯 PRONUNCIATION ASSESSMENT RESULTS\n"
                f"{'='*60}\n\n"
                f"📊 Component Scores:\n"
                f"  🎵 MFCC Similarity:    {mfcc_score:.2f}%\n"
                f"  🎤 Pitch Similarity:   {pitch_score:.2f}%\n"
                f"  ⏱️ Duration Similarity: {duration_score:.2f}%\n"
                f"  🔊 Formant Similarity: {formant_score:.2f}%\n\n"
                f"{'='*60}\n"
                f"📈 Final Score: {final_score:.2f}%\n"
                f"🏆 Grade: {grade}\n"
                f"{'='*60}\n\n"
                f"💬 Feedback:\n{feedback}\n\n"
                f"📁 Reference: {os.path.basename(self.audio1)}\n"
                f"📁 Student: {os.path.basename(self.audio2)}"
            )
            
            self.result_label.config(text=result_text, justify="left", fg="#2c3e50")
            
            # Show visualizations
            self.show_waveform(audio1_y, audio1_sr, "Reference Audio")
            self.show_waveform(audio2_y, audio2_sr, "Student Audio")
            self.show_spectrogram(audio1_y, audio1_sr, "Reference Audio")
            self.show_spectrogram(audio2_y, audio2_sr, "Student Audio")
            self.show_pitch_plot(self.audio1, "Reference Audio")
            self.show_pitch_plot(self.audio2, "Student Audio")
            self.show_mfcc_plot(audio1_y, audio1_sr, "Reference Audio")
            self.show_mfcc_plot(audio2_y, audio2_sr, "Student Audio")
            
            # Show comparison plot
            self.show_comparison_plot(features1, features2)
            
            # Save result
            self.save_result(final_score, mfcc_score, pitch_score, duration_score, formant_score)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.result_label.config(text=f"❌ Error: {str(e)}", fg="#e74c3c")
    
    def show_waveform(self, y, sr, title):
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform - {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def show_spectrogram(self, y, sr, title):
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(label='dB')
        plt.title(f"Spectrogram - {title}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def show_pitch_plot(self, file_path, title):
        try:
            sound = parselmouth.Sound(file_path)
            pitch = sound.to_pitch()
            values = pitch.selected_array['frequency']
            
            plt.figure(figsize=(10, 3))
            plt.plot(values, 'b-', linewidth=2)
            plt.title(f"Pitch Contour - {title}", fontsize=14, fontweight='bold')
            plt.xlabel("Frame")
            plt.ylabel("Frequency (Hz)")
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(500, values.max() + 50) if len(values) > 0 else 500)
            plt.show()
        except Exception as e:
            print(f"Pitch plot error: {e}")
    
    def show_mfcc_plot(self, y, sr, title):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar(label='MFCC Coefficient')
        plt.title(f"MFCC Plot - {title}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def show_comparison_plot(self, features1, features2):
        """
        Show comparison of features between two audio files
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Feature Comparison: Reference vs Student", fontsize=16, fontweight='bold')
        
        # MFCC comparison
        ax1 = axes[0, 0]
        mfcc1_mean = np.mean(features1['mfcc'], axis=1)
        mfcc2_mean = np.mean(features2['mfcc'], axis=1)
        ax1.plot(mfcc1_mean, 'b-', label='Reference', linewidth=2, alpha=0.7)
        ax1.plot(mfcc2_mean, 'r-', label='Student', linewidth=2, alpha=0.7)
        ax1.fill_between(range(len(mfcc1_mean)), mfcc1_mean, mfcc2_mean, alpha=0.2, color='purple')
        ax1.set_title('MFCC Comparison', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Coefficient')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pitch distribution
        ax2 = axes[0, 1]
        pitch1 = features1['pitch_contour']
        pitch2 = features2['pitch_contour']
        pitch1_valid = pitch1[pitch1 > 0]
        pitch2_valid = pitch2[pitch2 > 0]
        if len(pitch1_valid) > 0 and len(pitch2_valid) > 0:
            ax2.hist(pitch1_valid, bins=20, alpha=0.5, label='Reference', color='blue')
            ax2.hist(pitch2_valid, bins=20, alpha=0.5, label='Student', color='red')
        ax2.set_title('Pitch Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Energy comparison
        ax3 = axes[1, 0]
        energy1 = features1['energy_contour']
        energy2 = features2['energy_contour']
        min_len = min(len(energy1), len(energy2))
        if min_len > 0:
            ax3.plot(energy1[:min_len], 'b-', label='Reference', linewidth=2, alpha=0.7)
            ax3.plot(energy2[:min_len], 'r-', label='Student', linewidth=2, alpha=0.7)
        ax3.set_title('Energy Comparison', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Energy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Formant comparison
        ax4 = axes[1, 1]
        formants1 = np.array(features1['formants'])
        formants2 = np.array(features2['formants'])
        x = np.arange(len(formants1))
        width = 0.35
        if len(formants1) > 0:
            ax4.bar(x - width/2, formants1, width, label='Reference', alpha=0.7, color='blue')
            ax4.bar(x + width/2, formants2, width, label='Student', alpha=0.7, color='red')
        ax4.set_title('Formant Comparison', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Formant')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['F1', 'F2', 'F3'])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def load_dataset(self):
        """Load dataset from specified path"""
        path = self.path_entry.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Dataset path '{path}' does not exist.")
            return
        
        try:
            self.dataset = ArabicDataset(path)
            self.experiments = Experiments(self.dataset)
            self.update_dataset_info()
            messagebox.showinfo("Success", f"Dataset loaded successfully!\n\n"
                                          f"👥 Speakers: {self.dataset.get_speaker_count()}\n"
                                          f"📝 Words: {self.dataset.get_word_count()}\n"
                                          f"📊 Total Files: {self.dataset.get_speaker_count() * self.dataset.get_word_count()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def refresh_dataset(self):
        """Refresh dataset information"""
        if self.dataset:
            self.update_dataset_info()
            messagebox.showinfo("Success", "Dataset refreshed!")
        else:
            messagebox.showwarning("Warning", "No dataset loaded. Please load a dataset first.")
    
    def update_dataset_info(self):
        """Update dataset information in the GUI"""
        if not self.dataset:
            return
        
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Update treeview with dataset files
        for speaker in self.dataset.speakers:
            words = self.dataset.audio_files[speaker]
            for word, file_path in words.items():
                arabic = self.dataset.get_arabic_name(word)
                self.tree.insert('', 'end', values=(speaker, f"{word} ({arabic})", file_path))
        
        # Update info label
        info_text = (
            f"✅ Dataset loaded successfully!\n\n"
            f"📁 Path: {self.dataset.dataset_path}\n"
            f"👥 Number of Speakers: {self.dataset.get_speaker_count()}\n"
            f"📝 Number of Words: {self.dataset.get_word_count()}\n"
            f"📊 Total Audio Files: {self.dataset.get_speaker_count() * self.dataset.get_word_count()}\n\n"
            f"📋 Word List:\n"
        )
        
        for word in self.dataset.TEST_WORDS:
            arabic = self.dataset.get_arabic_name(word)
            english = self.dataset.get_english_meaning(word)
            sounds = ', '.join(self.dataset.get_word_sounds(word))
            info_text += f"  • {word} ({arabic}) - {english} - Sounds: {sounds}\n"
        
        info_text += f"\n🔊 Sound Types:\n"
        for sound_type, sounds in self.dataset.SOUND_TYPES.items():
            sound_examples = [s for s in sounds[:5]]
            info_text += f"  • {sound_type}: {', '.join(sound_examples)}...\n"
        
        self.dataset_info.config(text=info_text, fg="#2c3e50")
    
    def show_dataset_info(self):
        """Show detailed dataset information in a new window"""
        if not self.dataset:
            messagebox.showwarning("Warning", "No dataset loaded.")
            return
        
        # Create new window
        info_window = tk.Toplevel(self.root)
        info_window.title("Dataset Information")
        info_window.geometry("700x600")
        
        # Text widget with scrollbar
        text_widget = tk.Text(info_window, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = tk.Scrollbar(info_window, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        
        # Get dataset info as string
        info_str = StringIO()
        old_stdout = sys.stdout
        sys.stdout = info_str
        self.dataset.print_dataset_info()
        sys.stdout = old_stdout
        
        text_widget.insert(tk.END, info_str.getvalue())
        text_widget.config(state=tk.DISABLED)
    
    def export_dataset_info(self):
        """Export dataset information to a text file"""
        if not self.dataset:
            messagebox.showwarning("Warning", "No dataset loaded.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                info_str = StringIO()
                old_stdout = sys.stdout
                sys.stdout = info_str
                self.dataset.print_dataset_info()
                sys.stdout = old_stdout
                f.write(info_str.getvalue())
            
            messagebox.showinfo("Success", f"Dataset info exported to:\n{file_path}")
    
    def play_selected_audio(self, event=None):
        """Play selected audio from the treeview"""
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select an audio file.")
            return
        
        item = self.tree.item(selected[0])
        file_path = item['values'][2]
        
        if os.path.exists(file_path):
            try:
                import sounddevice as sd
                import soundfile as sf
                
                data, fs = sf.read(file_path)
                sd.play(data, fs)
                messagebox.showinfo("Playing", f"Playing: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play audio: {str(e)}")
        else:
            messagebox.showerror("Error", f"File not found: {file_path}")
    
    def save_result(self, final_score, mfcc_score, pitch_score, duration_score, formant_score):
        """Save assessment result to results archive"""
        from datetime import datetime
        
        if not os.path.exists("results"):
            os.makedirs("results")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to results file
        with open("results/assessment_results.txt", 'a', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Reference: {os.path.basename(self.audio1)}\n")
            f.write(f"Student: {os.path.basename(self.audio2)}\n")
            f.write(f"MFCC Score: {mfcc_score:.2f}%\n")
            f.write(f"Pitch Score: {pitch_score:.2f}%\n")
            f.write(f"Duration Score: {duration_score:.2f}%\n")
            f.write(f"Formant Score: {formant_score:.2f}%\n")
            f.write(f"Final Score: {final_score:.2f}%\n")
            f.write(f"{'='*60}\n\n")
        
        # Add to results tree
        self.results_tree.insert('', 0, values=(
            timestamp,
            "Pronunciation Assessment",
            f"{final_score:.2f}%",
            f"MFCC: {mfcc_score:.2f}%, Pitch: {pitch_score:.2f}%"
        ))
    
    def view_result(self):
        """View selected result"""
        selected = self.results_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a result.")
            return
        
        item = self.results_tree.item(selected[0])
        values = item['values']
        
        messagebox.showinfo("Result Details", 
                            f"📊 Result Details\n\n"
                            f"📅 Date: {values[0]}\n"
                            f"📝 Type: {values[1]}\n"
                            f"📈 Score: {values[2]}\n"
                            f"📋 Details: {values[3]}")
    
    def run_experiment_1(self):
        """Run same speaker experiment"""
        if not self.experiments:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        self.experiment_results.config(state=tk.NORMAL)
        self.experiment_results.delete(1.0, tk.END)
        self.experiment_results.insert(tk.END, "🔬 Running Experiment 1: Same Speaker vs Same Speaker\n")
        self.experiment_results.insert(tk.END, "="*60 + "\n\n")
        
        # Redirect print output to text widget
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.experiments.experiment_1_same_speaker()
            output = sys.stdout.getvalue()
            self.experiment_results.insert(tk.END, output)
            
            # Generate plots
            self.experiments.plot_results()
            
        except Exception as e:
            self.experiment_results.insert(tk.END, f"\n❌ Error: {str(e)}")
        
        sys.stdout = old_stdout
        self.experiment_results.see(tk.END)
        self.experiment_results.config(state=tk.DISABLED)
    
    def run_experiment_2(self):
        """Run different speakers experiment"""
        if not self.experiments:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        self.experiment_results.config(state=tk.NORMAL)
        self.experiment_results.delete(1.0, tk.END)
        self.experiment_results.insert(tk.END, "🔬 Running Experiment 2: Different Speakers\n")
        self.experiment_results.insert(tk.END, "="*60 + "\n\n")
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.experiments.experiment_2_different_speakers()
            output = sys.stdout.getvalue()
            self.experiment_results.insert(tk.END, output)
            
            # Generate plots
            self.experiments.plot_results()
            
        except Exception as e:
            self.experiment_results.insert(tk.END, f"\n❌ Error: {str(e)}")
        
        sys.stdout = old_stdout
        self.experiment_results.see(tk.END)
        self.experiment_results.config(state=tk.DISABLED)
    
    def run_experiment_3(self):
        """Run correct vs incorrect experiment"""
        if not self.experiments:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        self.experiment_results.config(state=tk.NORMAL)
        self.experiment_results.delete(1.0, tk.END)
        self.experiment_results.insert(tk.END, "🔬 Running Experiment 3: Correct vs Incorrect\n")
        self.experiment_results.insert(tk.END, "="*60 + "\n\n")
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.experiments.experiment_3_correct_vs_incorrect()
            output = sys.stdout.getvalue()
            self.experiment_results.insert(tk.END, output)
            
            # Generate plots
            self.experiments.plot_results()
            
        except Exception as e:
            self.experiment_results.insert(tk.END, f"\n❌ Error: {str(e)}")
        
        sys.stdout = old_stdout
        self.experiment_results.see(tk.END)
        self.experiment_results.config(state=tk.DISABLED)
    
    def run_all_experiments(self):
        """Run all experiments"""
        if not self.experiments:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        self.experiment_results.config(state=tk.NORMAL)
        self.experiment_results.delete(1.0, tk.END)
        self.experiment_results.insert(tk.END, "🚀 RUNNING ALL EXPERIMENTS\n")
        self.experiment_results.insert(tk.END, "="*60 + "\n\n")
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            self.experiments.run_all_experiments()
            output = sys.stdout.getvalue()
            self.experiment_results.insert(tk.END, output)
            
            # Generate plots
            self.experiments.plot_results()
            self.experiments.plot_similarity_matrix()
            
        except Exception as e:
            self.experiment_results.insert(tk.END, f"\n❌ Error: {str(e)}")
        
        sys.stdout = old_stdout
        self.experiment_results.see(tk.END)
        self.experiment_results.config(state=tk.DISABLED)