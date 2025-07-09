#!/usr/bin/env python3

import os

def is_csv_or_output_csv(filename: str) -> bool:
    return filename.endswith('.out.csv')

def clean_csv_files(root='.'):
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if is_csv_or_output_csv(file):
                file_path = os.path.join(dirpath, file)
                print(f"Deleting: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    clean_csv_files()
