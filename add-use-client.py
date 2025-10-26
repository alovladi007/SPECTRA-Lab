#!/usr/bin/env python3
import os
import glob

# Find all page.tsx files in dashboard
root_dir = "/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab/apps/web/src/app/dashboard"
files = glob.glob(f"{root_dir}/**/page.tsx", recursive=True)

print(f"Found {len(files)} files to check")

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file uses React hooks
    uses_hooks = ('useState' in content or 'useEffect' in content or
                  'useCallback' in content or 'useMemo' in content or
                  'useRef' in content or 'useContext' in content)

    # Check if already has 'use client'
    has_use_client = content.strip().startswith("'use client'") or content.strip().startswith('"use client"')

    if uses_hooks and not has_use_client:
        print(f"Adding 'use client' to: {filepath}")
        # Add 'use client' at the top
        new_content = "'use client'\n\n" + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

print("✓ All files fixed!")
