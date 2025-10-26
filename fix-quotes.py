#!/usr/bin/env python3
import os
import glob

# Find all page.tsx files
root_dir = "/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab/apps/web/src/app/dashboard"
files = glob.glob(f"{root_dir}/**/page.tsx", recursive=True)

print(f"Found {len(files)} files to fix")

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file has smart quotes
    if '\u2018' in content or '\u2019' in content or '\u201c' in content or '\u201d' in content:
        print(f"Fixing: {filepath}")

        # Replace smart quotes with regular quotes
        content = content.replace('\u2018', "'")  # ' -> '
        content = content.replace('\u2019', "'")  # ' -> '
        content = content.replace('\u201c', '"')  # " -> "
        content = content.replace('\u201d', '"')  # " -> "

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

print("✓ All files fixed!")
