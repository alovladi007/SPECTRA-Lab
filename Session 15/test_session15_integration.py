"""
SESSION 15: LIMS/ELN & REPORTING - Integration Tests
===================================================

Comprehensive test suite for LIMS, ELN, SOP, and Reporting features.

Author: SemiconductorLab Platform Team
Date: October 26, 2025
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import zipfile

# Import implementation
from session15_lims_eln_complete_implementation import (
    Sample, SampleStatus, NotebookEntry, SOP, SOPStatus,
    CustodyLog, EntrySignature, SignatureType,
    generate_barcode, generate_qr_code,
    ReportGenerator, FAIRExporter,
    ReportTemplate, FAIRExportRequest
)


class TestSampleManagement:
    """Test sample lifecycle and tracking"""
    
    def test_sample_creation(self):
        """Test sample creation with auto-generated ID"""
        sample = Sample(
            sample_id='TEST-2025-001',
            project_id=1,
            material_type='silicon',
            sample_type='wafer',
            location='Shelf A1',
            status=SampleStatus.RECEIVED
        )
        
        assert sample.sample_id == 'TEST-2025-001'
        assert sample.status == SampleStatus.RECEIVED
    
    def test_barcode_generation(self):
        """Test barcode image generation"""
        barcode = generate_barcode('TEST-2025-001')
        
        assert barcode.startswith('data:image/png;base64,')
        assert len(barcode) > 100
    
    def test_qr_code_generation(self):
        """Test QR code with embedded data"""
        qr_code = generate_qr_code('TEST-2025-001', {
            'project_id': 1,
            'material': 'silicon'
        })
        
        assert qr_code.startswith('data:image/png;base64,')
        assert len(qr_code) > 100


class TestELN:
    """Test Electronic Lab Notebook features"""
    
    def test_entry_creation(self):
        """Test creating notebook entry"""
        entry = NotebookEntry(
            entry_id='ELN-20251026-1234',
            project_id=1,
            author_id=1,
            title='Four-Point Probe Experiment',
            content='<h1>Results</h1><p>Measured Rs = 45.2 Ω/sq</p>',
            content_format='html',
            linked_samples=[1, 2],
            linked_runs=[10, 11]
        )
        
        assert entry.title == 'Four-Point Probe Experiment'
        assert entry.content_format == 'html'
        assert len(entry.linked_samples) == 2


class TestReportGeneration:
    """Test PDF report generation"""
    
    def test_report_generator_initialization(self):
        """Test report generator setup"""
        generator = ReportGenerator()
        assert generator.styles is not None
        assert 'CustomTitle' in generator.styles
    
    def test_report_generation(self):
        """Test generating PDF report"""
        generator = ReportGenerator()
        
        run_data = {
            'run_id': 'RUN-2025-001',
            'method_name': 'four_point_probe',
            'sample_id': 'TEST-2025-001',
            'operator': 'John Doe',
            'timestamp': datetime.utcnow().isoformat(),
            'parameters': {
                'current': 1e-3,
                'contacts': 4
            },
            'results': {
                'metrics': {
                    'sheet_resistance': {
                        'value': 45.2,
                        'uncertainty': 0.5,
                        'units': 'Ω/sq'
                    }
                }
            },
            'key_metrics': [
                {
                    'name': 'Sheet Resistance',
                    'value': 45.2,
                    'units': 'Ω/sq',
                    'spec': '40-50 Ω/sq',
                    'pass': True
                }
            ],
            'instrument': {
                'model': 'Keithley 2400',
                'serial': 'K2400-12345'
            }
        }
        
        template = ReportTemplate(
            template_name='standard',
            title='Test Report',
            sections=['summary', 'methods', 'results'],
            include_plots=True
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            output_path = generator.generate_run_report(
                run_data,
                f.name,
                template
            )
            
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 1000


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
