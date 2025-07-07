#!/usr/bin/env python3

import os

# Directories to remove
cache_dirs = ['__pycache__', '.mypy_cache', '.pytest_cache']

def clean_caches(root='.'):
    for dirpath, dirnames, filenames in os.walk(root):
        for d in cache_dirs:
            if d in dirnames:
                path = os.path.join(dirpath, d)
                print(f"Removing: {path}")
                os.system(f"rm -rf '{path}'")

if __name__ == "__main__":
    clean_caches()
