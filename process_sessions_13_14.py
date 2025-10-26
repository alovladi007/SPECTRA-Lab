#!/usr/bin/env python3
"""
Process and integrate Session 13 (SPC Hub), Session 14 (ML/VM), and Session 14 Optimization
"""

import os
import shutil
from pathlib import Path

# Define source and destination mappings
FILE_MAPPINGS = {
    # Session 13: SPC Hub files
    'Session 13/session13_spc_complete_implementation.py':
        'services/analysis/app/methods/spc/spc_hub.py',
    'Session 13/session13_spc_ui_components.tsx':
        'apps/web/src/app/(dashboard)/spc/page.tsx',
    'Session 13/test_session13_integration.py':
        'tests/integration/test_session13_spc.py',
    'Session 13/deploy_session13.sh':
        'scripts/deploy_session13.sh',
    'Session 13/session13_complete_documentation.md':
        'docs/sessions/session13_spc_documentation.md',
    'Session 13/Session_13_Complete_Delivery_Package.md':
        'docs/sessions/session13_delivery.md',
    'Session 13/SESSION_13_README.md':
        'docs/sessions/session13_README.md',

    # Session 14: ML/VM Hub files
    'Session 14/session14_vm_ml_complete_implementation.py':
        'services/analysis/app/methods/ml/vm_ml_hub.py',
    'Session 14/session14_vm_ml_ui_components.tsx':
        'apps/web/src/app/(dashboard)/ml/vm-models/page.tsx',
    'Session 14/session14_vm_ml_ui_components_part2.tsx':
        'apps/web/src/app/(dashboard)/ml/monitoring/page.tsx',
    'Session 14/test_session14_integration.py':
        'tests/integration/test_session14_ml_vm.py',
    'Session 14/deploy_session14.sh':
        'scripts/deploy_session14.sh',
    'Session 14/SESSION_14_README.md':
        'docs/sessions/session14_README.md',
    'Session 14/Session_14_Complete_Delivery_Package.md':
        'docs/sessions/session14_delivery.md',

    # Session 14 Optimization files
    'Session 14 Optimization/session14_enhanced_implementation.py':
        'services/analysis/app/methods/ml/enhanced_ml.py',
    'Session 14 Optimization/session14_enhanced_part2.py':
        'services/analysis/app/methods/ml/enhanced_ml_part2.py',
    'Session 14 Optimization/deploy_session14_enhanced.sh':
        'scripts/deploy_session14_enhanced.sh',
    'Session 14 Optimization/SESSION_14_ENHANCED_README.md':
        'docs/sessions/session14_enhanced_README.md',
    'Session 14 Optimization/SESSION_14_ENHANCEMENT_SUMMARY.md':
        'docs/sessions/session14_enhancement_summary.md',
    'Session 14 Optimization/SESSION_14_MASTER_DELIVERY.md':
        'docs/sessions/session14_master_delivery.md',
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
        if line.strip() in ['```', '```python', '```py', '```typescript', '```tsx', '```bash', '```sh', '```yaml', '```json']:
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
    print("SPECTRA-Lab Session 13 (SPC) & Session 14 (ML/VM + Enhanced) Integration")
    print("=" * 70)
    print()

    count = process_files()

    print()
    print("=" * 70)
    print(f"Integration complete! {count} files processed.")
    print("=" * 70)
