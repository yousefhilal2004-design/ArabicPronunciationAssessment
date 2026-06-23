"""
main.py
=======
Entry point. Two modes:
  python main.py            -> launches the GUI (manual reference-vs-student comparison)
  python main.py --experiments -> loads the dataset and runs all 5 experiments
                                   headlessly (recommended for generating report numbers)
"""

import os
import sys
import tkinter as tk

from dataset import ArabicDataset
from experiments import Experiments


def create_results_folder():
    if not os.path.exists("results"):
        os.makedirs("results")


def load_dataset(dataset_path="./dataset"):
    if os.path.exists(dataset_path):
        print("Loading dataset...")
        dataset = ArabicDataset(dataset_path)
        dataset.print_dataset_info()
        return dataset
    else:
        print(f"Dataset path '{dataset_path}' not found!")
        print("Expected structure:")
        print("  dataset/")
        print("    Speaker_1/")
        print("      arabi_1.wav, daw_1.wav, ghorfa_1.wav, ...")
        print("    Speaker_2/")
        print("      ...")
        return None


def run_experiments_headless():
    create_results_folder()
    dataset = load_dataset()
    if dataset is None:
        sys.exit(1)
    if dataset.get_speaker_count() == 0:
        print("No speakers found - check dataset folder naming (Speaker_1..Speaker_5).")
        sys.exit(1)

    exp = Experiments(dataset)
    exp.run_all_experiments()

    # Save numeric results to CSV for the report
    import pandas as pd
    for key, results in exp.results.items():
        if results:
            pd.DataFrame(results).to_csv(f"results/{key}.csv", index=False)
            print(f"Saved results/{key}.csv")


def run_gui():
    from gui import PronunciationGUI
    create_results_folder()
    dataset = load_dataset()

    root = tk.Tk()
    app = PronunciationGUI(root)

    if dataset:
        app.dataset = dataset
        app.experiments = Experiments(dataset)
        app.update_dataset_info()

    root.mainloop()


def main():
    if "--experiments" in sys.argv:
        run_experiments_headless()
    else:
        run_gui()


if __name__ == "__main__":
    main()