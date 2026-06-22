import os
import tkinter as tk
from gui import PronunciationGUI
from dataset import ArabicDataset
from experiments import Experiments


def create_results_folder():
    """Create results folder if it doesn't exist"""
    if not os.path.exists("results"):
        os.makedirs("results")


def load_dataset():
    """Load the dataset from the dataset folder"""
    dataset_path = "./dataset"
    
    if os.path.exists(dataset_path):
        print("Loading dataset...")
        dataset = ArabicDataset(dataset_path)
        dataset.print_dataset_info()
        return dataset
    else:
        print(f"❌ Dataset path '{dataset_path}' not found!")
        print("Please make sure your dataset is in the following structure:")
        print("  dataset/")
        print("    speaker_1/")
        print("      arabi_1.wav, daw_1.wav, ghorfa_1.wav, ...")
        print("    speaker_2/")
        print("      arabi_2.wav, daw_2.wav, ghorfa_2.wav, ...")
        print("    ...")
        return None


def main():
    """Main entry point"""
    # Create necessary folders
    create_results_folder()
    
    # Load dataset
    dataset = load_dataset()
    
    # Launch GUI
    root = tk.Tk()
    app = PronunciationGUI(root)
    
    # Pre-load the dataset
    if dataset:
        app.dataset = dataset
        app.experiments = Experiments(dataset) if dataset else None
        
        # Update dataset info in GUI
        app.update_dataset_info()
    
    root.mainloop()


if __name__ == "__main__":
    main()