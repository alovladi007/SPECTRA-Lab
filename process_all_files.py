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

    # Session 6 - Additional Files
    'Session_6__Electrical_III_-_DLTS__EBIC__PCD_Implementation': 'docs/sessions/session6_dlts_ebic_pcd.md',
    'Session_6__Electrical_III_-_Complete_Backend_Analysis_Modules': 'services/analysis/app/methods/electrical/session6_advanced_modules.py',
    'Session_6__Electrical_III_-_UI_Components__Continued_': 'apps/web/src/app/(dashboard)/electrical/advanced/page.tsx',
    'Complete_Session_6__Electrical_III_-_UI_Components': 'apps/web/src/app/(dashboard)/electrical/session6-components.tsx',
    'Session_6__Complete_Integration_Tests': 'tests/integration/test_session6_complete.py',
    'Session_6__Electrical_III_-_Master_Deployment_Script': 'scripts/deploy_session6.sh',
    'Session_6__Electrical_III_-_Complete_Delivery_Package': 'docs/sessions/session6_delivery.md',
    '_Session_6__Electrical_III_-_Complete_Documentation': 'docs/sessions/session6_documentation.md',
    'Test_Data_Generators_for_Session_6__Electrical_III__DLTS__EBIC__PCD_': 'scripts/dev/generate_session6_test_data.py',

    # Session 5 - Additional Files
    'Session_5_complete_I-C_I-V_characterization_': 'docs/sessions/session5_iv_cv_characterization.md',
    'Session_5_Complete_Delivery_Package': 'docs/sessions/session5_delivery.md',
    'Session_5_complete_deliverables_': 'docs/sessions/session5_deliverables.md',
    'Session_5_complete_delivery_package_': 'docs/sessions/session5_delivery_package.md',
    'Complete_Session_5__Electrical_II_-_UI_Components': 'apps/web/src/app/(dashboard)/electrical/session5-components.tsx',
    'Session_5_I-V_and_C-V_test_data_': 'data/test_data/session5_iv_cv_data.json',
    'SemiconductorLab_Platform_-_Session_5_Deployment_Script': 'scripts/deploy_session5.sh',
    'Test_Session_5__Integration_Test': 'tests/integration/test_session5_integration.py',
    'Session_5_Intégration_test': 'tests/integration/session5_integration_test.py',

    # UI Shell & Components
    'Next.js UI shell': 'apps/web/src/components/layout/AppShell.tsx',
    'MOSFET_characterization_UI': 'apps/web/src/app/(dashboard)/electrical/mosfet-advanced/page.tsx',
    'BJT_characterization_': 'apps/web/src/app/(dashboard)/electrical/bjt-advanced/page.tsx',
    'MOSFET_Characterization': 'docs/methods/mosfet_characterization.md',
    'MOSFET_and_solar_Cell': 'docs/methods/mosfet_solar_cell.md',
    'C-V_Profiling_': 'docs/methods/cv_profiling.md',
    'Capacitance-Voltage__C-V__Profiling_Analysis': 'docs/methods/cv_profiling_analysis.md',

    # Status Reports & Roadmaps
    'SemiconductorLab_Platform_-_Complete_Implementation_Roadmap___Status_Report': 'docs/status/implementation_roadmap.md',
    'SemiconductorLab_Platform_-_Complete_Status_Report': 'docs/status/complete_status_report.md',
    'Complete_implementation_summary_': 'docs/status/implementation_summary.md',
    'Comprehensive_Code_Review___Production_Roadmap': 'docs/status/code_review_roadmap.md',

    # Additional Tests & Implementation
    'Complete_integration_test': 'tests/integration/complete_integration_test.py',
    'Advanced_Test_Cases___Validation_Scenarios_': 'tests/validation/advanced_test_cases.py',
    'Complete_Implementation_': 'docs/implementation/complete_implementation.md',

    # Additional Deployment & Infrastructure
    'Deployment_and_guide_start_': 'docs/deployment/deployment_guide.md',
    'Plugin_Architectures_': 'docs/architecture/plugin_architecture.md',
    'Complete_Database_': 'db/migrations/002_complete_schema.sql',

    # Optical Session (Session 7)
    'Complete Optical platform': 'docs/sessions/session7_optical_platform.md',
    'VISA_SCPI': 'docs/architecture/visa_scpi_protocol.md',
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

def process_session7_directory(source_dir: Path, base_dir: Path):
    """Process Session 7 directory structure"""
    session7_dirs = [
        'Session7_Optical_I_Complete_Package_20251024_034815/Session7_Optical_I',
        'session7_optical_complete (1)'
    ]

    processed = 0
    for dir_name in session7_dirs:
        session7_path = source_dir / dir_name
        if not session7_path.exists():
            continue

        # Process Session 7 files
        for item in session7_path.rglob('*'):
            if item.is_file() and not item.name.startswith('.'):
                relative_path = item.relative_to(session7_path)

                # Determine destination based on file type
                if item.suffix == '.py':
                    if 'test' in item.name:
                        dest = base_dir / 'tests' / 'integration' / f'session7_{item.name}'
                    elif 'analyzer' in item.name:
                        dest = base_dir / 'services' / 'analysis' / 'app' / 'methods' / 'optical' / item.name
                    else:
                        dest = base_dir / 'scripts' / item.name
                elif item.suffix in ['.tsx', '.jsx']:
                    dest = base_dir / 'apps' / 'web' / 'src' / 'app' / '(dashboard)' / 'optical' / item.name
                elif item.suffix == '.md':
                    dest = base_dir / 'docs' / 'sessions' / f'session7_{item.name}'
                elif item.suffix == '.sh':
                    dest = base_dir / 'scripts' / item.name
                else:
                    dest = base_dir / 'docs' / 'sessions' / 'session7' / str(relative_path)

                try:
                    process_file(item, dest)
                    processed += 1
                except Exception as e:
                    print(f"  ✗ Error processing {item.name}: {e}")

    return processed

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
    print(f"Processing {len(FILE_MAPPINGS)} mapped files + Session 7 directories...\n")

    processed = 0
    errors = []

    # Process mapped files
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

    # Process Session 7 directories
    print("\nProcessing Session 7 directories...")
    session7_count = process_session7_directory(source_dir, base_dir)
    processed += session7_count
    print(f"Processed {session7_count} Session 7 files")

    print("\n" + "=" * 80)
    print(f"Processing complete: {processed} files processed")

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ All files processed successfully!")

    print("=" * 80)

if __name__ == '__main__':
    main()
