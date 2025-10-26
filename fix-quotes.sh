#!/bin/bash

# Replace smart quotes with regular quotes in all dashboard page.tsx files

cd "/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab/apps/web"

find src/app/dashboard -name "page.tsx" -type f | while read -r file; do
  echo "Fixing: $file"
  # Replace left single quote (') with regular apostrophe (')
  perl -i -pe 's/\x{2018}/'"'"'/g' "$file"
  # Replace right single quote (') with regular apostrophe (')
  perl -i -pe 's/\x{2019}/'"'"'/g' "$file"
  # Replace left double quote (") with regular quote (")
  perl -i -pe 's/\x{201C}/"/g' "$file"
  # Replace right double quote (") with regular quote (")
  perl -i -pe 's/\x{201D}/"/g' "$file"
done

echo "✓ All files fixed!"
