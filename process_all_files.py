#!/usr/bin/env python3
"""
Automated File Processor for SPECTRA-Lab
Processes all files from semiconductorlab_all_project_files/ directory
"""

import os
import re
from pathlib import Path

# File mapping - maps source files to their correct destination
FILE_MAPPINGS = {
    # Core infrastructure
    'SQL_Alchemy_': 'services/instruments/app/models/__init__.py',
    'S3_Complete_Visa': 'services/instruments/app/drivers/core/connection.py',
    'S3_Complete_Plugin_Architecture_': 'services/instruments/app/drivers/core/plugin_manager.py',
    'Keithley_2300': 'services/instruments/app/drivers/builtin/keithley_2400.py',
    'Ocean_Optics_Spectrometer_': 'services/instruments/app/drivers/builtin/oceanoptics_spectrometer.py',

    # Pydantic schemas
    'Pydantic_schemas': 'src/backend/models/pydantic_schemas.py',

    # Reference drivers
    'Reference_Drivers_': 'src/drivers/instruments/reference_drivers.py',
    'HL_simulator_Framework_': 'src/drivers/simulators/hil_framework.py',

    # Test data generators
    'Electrical_test_data_generators': 'scripts/dev/generate_electrical_test_data.py',
    'Test_data_generation_': 'scripts/generate_test_data.py',
    'Session_5_test_data_generators': 'scripts/dev/generate_session5_test_data.py',

    # Analysis modules - Electrical
    '4_point_probe_analysis_': 'services/analysis/app/methods/electrical/four_point_probe.py',
    'Hall_effects_analysis_module_': 'services/analysis/app/methods/electrical/hall_effect.py',
    'I-V_characterization_': 'services/analysis/app/methods/electrical/iv_characterization.py',
    'C-V_profiling_analysis_': 'services/analysis/app/methods/electrical/cv_profiling.py',
    'BJT_I-V_Analysis_': 'services/analysis/app/methods/electrical/bjt_analysis.py',
    'MOSFET_I-V_analysis_': 'services/analysis/app/methods/electrical/mosfet_analysis.py',
    'Solar_Cell_I-V_analysis_': 'services/analysis/app/methods/electrical/solar_cell_analysis.py',
    'DLTS_Analysis_': 'services/analysis/app/methods/electrical/dlts_analysis.py',

    # UI Components
    '4_point_probe': 'apps/web/src/app/(dashboard)/electrical/four-point-probe/page.tsx',
    'Hall_effects_measurements_UI': 'apps/web/src/app/(dashboard)/electrical/hall-effect/page.tsx',
    'BJT_Charachterization_Interface': 'apps/web/src/app/(dashboard)/electrical/bjt/page.tsx',
    'MOSFET_Characterization_Interface': 'apps/web/src/app/(dashboard)/electrical/mosfet/page.tsx',
    'Solar_Cell_Charachterization_UI': 'apps/web/src/app/(dashboard)/electrical/solar-cell/page.tsx',
    'C-V_Profiling_Interface': 'apps/web/src/app/(dashboard)/electrical/cv-profiling/page.tsx',

    # Integration Tests
    'Complete_Integration_Test_Suite_for_Session_5__Electrical_II': 'tests/integration/test_session5_electrical_workflows.py',
    'Complete_Integration_Test_Suite_for_Session_6__Electrical_III': 'tests/integration/test_session6_electrical_workflows.py',
    'S3_complete_Test_Suite_': 'tests/unit/test_core_infrastructure.py',

    # Utilities
    'Unit_handling_system_': 'packages/common/semiconductorlab_common/units.py',
    'Object_storage_': 'src/backend/services/storage.py',

    # Documentation
    'Session_1__Program_setup_and_Architecture_': 'docs/sessions/session1_setup_architecture.md',
    'Session_1_an_2_complete_': 'docs/sessions/sessions_1_2_complete.md',
    'Session_4_electrical_1_complete_': 'docs/sessions/session4_complete.md',
    'Session_5__Electrical_II_-_Complete_Implementation_Guide': 'docs/sessions/session5_complete.md',
    'Session_6__Electrical_III_-_Complete_Implementation_Package': 'docs/sessions/session6_complete.md',
    'Session_5_Method_playbooks': 'docs/methods/session5_playbooks.md',
    'Requirement_Documents_': 'docs/REQUIREMENTS.md',
    'Lab_technician_training_guide_': 'docs/guides/technician_training.md',
    'Semiconductor_lab_-_Repository_Structure_': 'docs/REPOSITORY_STRUCTURE.md',

    # Deployment Scripts
    'Complete_Deployment_Script_': 'scripts/deploy_complete.sh',
    'Production_deployment_script_': 'scripts/deploy_production.sh',
    'SemiconductorLab_Platform_-_Master_Deployment_Script': 'scripts/deploy_master.sh',

    # Configuration
    'Docker_Compose_': 'infra/docker/docker-compose.yml',
    'Make_File': 'Makefile',
    'Open_API_specification_': 'docs/api/openapi_specification.yaml',
    'Database_schema_': 'db/migrations/001_initial_schema.sql',
    'Architecture_': 'docs/architecture/overview.md',
    'Data_model_and_specification_': 'docs/DATA_MODEL_SPECIFICATION.md',
    'Semiconductor_Lab_platform_': 'docs/ROADMAP.md',
}

def clean_python_content(content: str) -> str:
    """Clean file content by removing markdown artifacts"""
    lines = content.split('\n')
    cleaned_lines = []
    skip_next = False

    for line in lines:
        # Skip markdown code block markers
        if line.strip() in ['```', '```python', '```py', '```bash', '```sh', '```typescript', '```tsx', '```yaml', '```sql']:
            continue

        # Remove leading triple backticks with content
        if line.strip().startswith('```'):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def process_file(source_path: Path, dest_path: Path):
    """Process a single file"""
    print(f"Processing: {source_path.name} -> {dest_path}")

    # Read source
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Clean content based on file type
    if dest_path.suffix in ['.py', '.tsx', '.ts', '.sh', '.md', '.yaml', '.yml']:
        content = clean_python_content(content)

    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write destination
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Written to {dest_path}")

def main():
    """Main processing function"""
    base_dir = Path(__file__).parent
    source_dir = base_dir / 'semiconductorlab_all_project_files'

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    print("=" * 80)
    print("SPECTRA-Lab File Processor")
    print("=" * 80)
    print(f"\nSource directory: {source_dir}")
    print(f"Processing {len(FILE_MAPPINGS)} files...\n")

    processed = 0
    errors = []

    for source_name, dest_path in FILE_MAPPINGS.items():
        source_path = source_dir / source_name
        dest_full_path = base_dir / dest_path

        if not source_path.exists():
            errors.append(f"Source file not found: {source_name}")
            continue

        try:
            process_file(source_path, dest_full_path)
            processed += 1
        except Exception as e:
            errors.append(f"Error processing {source_name}: {e}")

    print("\n" + "=" * 80)
    print(f"Processing complete: {processed}/{len(FILE_MAPPINGS)} files processed")

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ All files processed successfully!")

    print("=" * 80)

if __name__ == '__main__':
    main()
