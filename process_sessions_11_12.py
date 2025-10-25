#!/usr/bin/env python3
"""
Process and integrate Session 11 (Complete Update) and Session 12 files
"""

import os
import shutil
from pathlib import Path

# Define source and destination mappings
FILE_MAPPINGS = {
    # Session 11 Complete Update files
    'Session 11 Complete Update/session11_surface_analysis_complete_implementation.py':
        'services/analysis/app/methods/chemical/surface_analysis.py',
    'Session 11 Complete Update/session11_surface_analysis_ui_components.tsx':
        'apps/web/src/app/(dashboard)/chemical/surface-analysis/page.tsx',
    'Session 11 Complete Update/test_session11_integration.py':
        'tests/integration/test_session11_surface_analysis.py',
    'Session 11 Complete Update/deploy_session11.sh':
        'scripts/deploy_session11_surface.sh',
    'Session 11 Complete Update/session11_complete_documentation.md':
        'docs/sessions/session11_surface_analysis_documentation.md',
    'Session 11 Complete Update/Session_11_Complete_Delivery_Package.md':
        'docs/sessions/session11_surface_delivery.md',
    'Session 11 Complete Update/SESSION_11_README.md':
        'docs/sessions/session11_surface_README.md',

    # Session 12 Completed files
    'Session 12 Completed/session12_chemical_bulk_complete_implementation.py':
        'services/analysis/app/methods/chemical/bulk_analysis.py',
    'Session 12 Completed/session12_chemical_bulk_ui_components.tsx':
        'apps/web/src/app/(dashboard)/chemical/bulk-analysis/page.tsx',
    'Session 12 Completed/test_session12_integration.py':
        'tests/integration/test_session12_bulk_analysis.py',
    'Session 12 Completed/deploy_session12.sh':
        'scripts/deploy_session12.sh',
    'Session 12 Completed/session12_complete_documentation.md':
        'docs/sessions/session12_bulk_documentation.md',
    'Session 12 Completed/Session_12_Complete_Delivery_Package.md':
        'docs/sessions/session12_delivery.md',
    'Session 12 Completed/SESSION_12_README.md':
        'docs/sessions/session12_README.md',
}

def clean_content(content: str, file_path: str) -> str:
    """Remove markdown artifacts from code files"""
    # Only clean code files, not documentation
    if file_path.endswith('.md'):
        return content

    lines = content.split('\n')
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        # Skip markdown code block markers
        if line.strip() in ['```', '```python', '```py', '```typescript', '```tsx', '```bash', '```sh']:
            in_code_block = not in_code_block
            continue
        if line.strip().startswith('```'):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def process_files():
    """Process all files and copy to correct locations"""
    downloads_dir = Path.home() / 'Downloads'
    repo_dir = Path('/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab')

    processed_count = 0

    for source_rel, dest_rel in FILE_MAPPINGS.items():
        source_path = downloads_dir / source_rel
        dest_path = repo_dir / dest_rel

        if not source_path.exists():
            print(f"⚠️  Source file not found: {source_path}")
            continue

        # Create destination directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Read, clean, and write content
        print(f"Processing: {source_path.name}")
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()

        cleaned_content = clean_content(content, str(dest_path))

        with open(dest_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        # Set executable permission for shell scripts
        if dest_path.suffix == '.sh':
            os.chmod(dest_path, 0o755)
            print(f"  ✓ Set executable: {dest_path.name}")

        processed_count += 1
        print(f"  ✓ Created: {dest_path}")

    print(f"\n✅ Successfully processed {processed_count} files")
    return processed_count

if __name__ == '__main__':
    print("=" * 70)
    print("SPECTRA-Lab Session 11 (Update) & Session 12 Integration")
    print("=" * 70)
    print()

    count = process_files()

    print()
    print("=" * 70)
    print(f"Integration complete! {count} files processed.")
    print("=" * 70)
