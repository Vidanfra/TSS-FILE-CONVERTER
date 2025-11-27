import tkinter as tk
from tkinter import filedialog
import os
    
def list_files(start_path, suffixes):
    matched_files = []
    for root, _, filenames in os.walk(start_path):  # Avoid shadowing 'files' variable
        for file in filenames:
            if file.endswith(suffixes):  # Strict suffix matching
                matched_files.append(os.path.normpath(os.path.join(root, file)))
    
    return matched_files

def get_files():
    folder_path = filedialog.askdirectory(title="Select a folder")
    files = list_files(folder_path, ("_AF.txt", "_AL.txt"))
    return files

if __name__ == "__main__":
    files = get_files()
    print("Files found:")
    for i, file in enumerate(files):
        print(f"Target {i} - File: {file}")


