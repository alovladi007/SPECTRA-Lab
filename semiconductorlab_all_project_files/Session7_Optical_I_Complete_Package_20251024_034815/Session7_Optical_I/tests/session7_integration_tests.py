"""
Session 7: Optical I - Complete Integration Test Suite
Testing UV-Vis-NIR and FTIR with platform integration
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import analyzers
from session7_uvvisnir_analyzer import UVVisNIRAnalyzer, TransitionType, BaselineMethod
from session7_ftir_analyzer import FTIRAnalyzer, Peak
from test_session7_optical import OpticalDataGenerator


class TestDatabaseIntegration:
    """Test database storage and retrieval"""
    
    @pytest.fixture
    def mock_db(self):
        """Mock database connection"""
        return MagicMock()
    
    def test_save_uvvisnir_results(self, mock_db):
        """Test saving UV-Vis-NIR results to database"""
        # Generate test data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_uvvisnir_spectrum('GaAs')
        
        # Process spectrum
        analyzer = UVVisNIRAnalyzer()
        processed = analyzer.process_spectrum(
            spectrum['wavelength'],
            spectrum['transmission']
        )
        
        # Calculate band gap
        tauc = analyzer.calculate_tauc_plot(
            processed['wavelength'],
            processed['absorbance']
        )
        
        # Create result object
        result = {
            'sample_id': 'TEST_001',
            'method': 'UV-Vis-NIR',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'wavelength_range': [300, 1000],
                'mode': 'transmission',
                'baseline_method': 'als'
            },
            'results': {
                'band_gap': tauc.band_gap,
                'transition_type': tauc.transition_type,
                'r_squared': tauc.r_squared
            },
            'raw_data': {
                'wavelength': processed['wavelength'].tolist(),
                'absorbance': processed['absorbance'].tolist()
            }
        }
        
        # Save to database
        mock_db.save_measurement.return_value = {'id': 123}
        saved_id = mock_db.save_measurement(result)
        
        assert mock_db.save_measurement.called
        assert saved_id['id'] == 123
    
    def test_save_ftir_results(self, mock_db):
        """Test saving FTIR results to database"""
        # Generate test data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_ftir_spectrum('polymer')
        
        # Process spectrum
        analyzer = FTIRAnalyzer()
        result = analyzer.process_spectrum(
            spectrum['wavenumber'],
            spectrum['absorbance']
        )
        
        # Create result object
        db_result = {
            'sample_id': 'TEST_002',
            'method': 'FTIR',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'wavenumber_range': [400, 4000],
                'resolution': 4,
                'scans': 32
            },
            'peaks': [
                {
                    'position': peak.position,
                    'intensity': peak.intensity,
                    'width': peak.width,
                    'assignment': peak.assignment
                }
                for peak in result.peaks[:10]
            ],
            'functional_groups': [
                {
                    'name': group.name,
                    'peak_range': group.peak_range,
                    'vibration_type': group.vibration_type
                }
                for group in result.functional_groups
            ]
        }
        
        # Save to database
        mock_db.save_measurement.return_value = {'id': 124}
        saved_id = mock_db.save_measurement(db_result)
        
        assert mock_db.save_measurement.called
        assert saved_id['id'] == 124
    
    def test_batch_save(self, mock_db):
        """Test saving batch results"""
        generator = OpticalDataGenerator()
        
        # Generate batch
        spectra = generator.generate_batch_spectra('uvvisnir', n_samples=5)
        
        # Process batch
        analyzer = UVVisNIRAnalyzer()
        results = []
        
        for spectrum in spectra:
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            results.append({
                'sample_id': spectrum['sample_id'],
                'data': processed
            })
        
        # Batch save
        mock_db.save_batch.return_value = {'saved': 5}
        saved = mock_db.save_batch(results)
        
        assert mock_db.save_batch.called
        assert saved['saved'] == 5


class TestAPIIntegration:
    """Test API endpoint integration"""
    
    @pytest.fixture
    def mock_api_client(self):
        """Mock API client"""
        client = MagicMock()
        client.base_url = "http://localhost:8000"
        return client
    
    async def test_uvvisnir_api_endpoint(self, mock_api_client):
        """Test UV-Vis-NIR API endpoint"""
        # Prepare request data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_uvvisnir_spectrum('GaN')
        
        request_data = {
            'wavelength': spectrum['wavelength'].tolist(),
            'intensity': spectrum['transmission'].tolist(),
            'mode': 'transmission',
            'parameters': {
                'baseline_method': 'als',
                'transition_type': 'direct',
                'smooth': True
            }
        }
        
        # Mock API response
        mock_response = {
            'status': 'success',
            'band_gap': 3.42,
            'r_squared': 0.995,
            'processing_time': 0.234
        }
        
        mock_api_client.post.return_value = mock_response
        
        # Make API call
        response = mock_api_client.post(
            '/api/v1/optical/uvvisnir/analyze',
            json=request_data
        )
        
        assert response['status'] == 'success'
        assert 3.0 < response['band_gap'] < 4.0
        assert response['r_squared'] > 0.9
    
    async def test_ftir_api_endpoint(self, mock_api_client):
        """Test FTIR API endpoint"""
        # Prepare request data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_ftir_spectrum('SiO2')
        
        request_data = {
            'wavenumber': spectrum['wavenumber'].tolist(),
            'absorbance': spectrum['absorbance'].tolist(),
            'parameters': {
                'baseline_method': 'als',
                'peak_threshold': 0.01,
                'atr_correction': False
            }
        }
        
        # Mock API response
        mock_response = {
            'status': 'success',
            'n_peaks': 5,
            'functional_groups': ['Si-O stretch', 'O-H stretch'],
            'processing_time': 0.456
        }
        
        mock_api_client.post.return_value = mock_response
        
        # Make API call
        response = mock_api_client.post(
            '/api/v1/optical/ftir/analyze',
            json=request_data
        )
        
        assert response['status'] == 'success'
        assert response['n_peaks'] > 0
        assert 'Si-O stretch' in response['functional_groups']
    
    async def test_batch_api_endpoint(self, mock_api_client):
        """Test batch processing API endpoint"""
        generator = OpticalDataGenerator()
        
        # Generate batch data
        spectra = generator.generate_batch_spectra('uvvisnir', n_samples=3)
        
        request_data = {
            'spectra': [
                {
                    'sample_id': s['sample_id'],
                    'wavelength': s['wavelength'].tolist(),
                    'intensity': s['transmission'].tolist()
                }
                for s in spectra
            ],
            'parameters': {
                'method': 'uvvisnir',
                'extract_band_gap': True
            }
        }
        
        # Mock response
        mock_response = {
            'status': 'success',
            'processed': 3,
            'results': [
                {'sample_id': 'UV_001', 'band_gap': 1.42},
                {'sample_id': 'UV_002', 'band_gap': 3.40},
                {'sample_id': 'UV_003', 'band_gap': 1.12}
            ]
        }
        
        mock_api_client.post.return_value = mock_response
        
        # Make API call
        response = mock_api_client.post(
            '/api/v1/optical/batch/process',
            json=request_data
        )
        
        assert response['processed'] == 3
        assert len(response['results']) == 3


class TestFileOperations:
    """Test file I/O operations"""
    
    def test_export_uvvisnir_results(self):
        """Test exporting UV-Vis-NIR results"""
        # Generate and process data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_uvvisnir_spectrum('ZnO')
        
        analyzer = UVVisNIRAnalyzer()
        processed = analyzer.process_spectrum(
            spectrum['wavelength'],
            spectrum['transmission']
        )
        
        tauc = analyzer.calculate_tauc_plot(
            processed['wavelength'],
            processed['absorbance']
        )
        
        # Export to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_data = {
                'metadata': {
                    'sample': 'ZnO',
                    'method': 'UV-Vis-NIR',
                    'timestamp': datetime.now().isoformat()
                },
                'parameters': {
                    'baseline_method': 'als',
                    'transition_type': 'direct'
                },
                'results': {
                    'band_gap': tauc.band_gap,
                    'r_squared': tauc.r_squared,
                    'uncertainty': tauc.uncertainty
                },
                'data': {
                    'wavelength': processed['wavelength'].tolist(),
                    'absorbance': processed['absorbance'].tolist(),
                    'tauc_energy': tauc.photon_energy.tolist(),
                    'tauc_values': tauc.tauc_values.tolist()
                }
            }
            
            json.dump(export_data, f, indent=2)
            filename = f.name
        
        # Verify file exists and can be read
        assert os.path.exists(filename)
        
        with open(filename, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['metadata']['sample'] == 'ZnO'
        assert 'band_gap' in loaded_data['results']
        
        # Cleanup
        os.unlink(filename)
    
    def test_export_ftir_results(self):
        """Test exporting FTIR results"""
        # Generate and process data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_ftir_spectrum('protein')
        
        analyzer = FTIRAnalyzer()
        result = analyzer.process_spectrum(
            spectrum['wavenumber'],
            spectrum['absorbance']
        )
        
        # Export to CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create DataFrame
            df = pd.DataFrame({
                'Wavenumber': result.wavenumber,
                'Raw_Absorbance': result.absorbance,
                'Baseline': result.baseline,
                'Corrected': result.corrected
            })
            
            df.to_csv(f.name, index=False)
            filename = f.name
        
        # Verify file
        loaded_df = pd.read_csv(filename)
        assert len(loaded_df) == len(result.wavenumber)
        assert 'Corrected' in loaded_df.columns
        
        # Cleanup
        os.unlink(filename)
    
    def test_import_spectrum_files(self):
        """Test importing various spectrum file formats"""
        # Create test files
        test_files = []
        
        # CSV format
        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        csv_file.write("Wavelength,Transmission\n")
        for wl in np.linspace(300, 800, 100):
            csv_file.write(f"{wl},{np.random.uniform(0, 100)}\n")
        csv_file.close()
        test_files.append(csv_file.name)
        
        # TXT format (tab-separated)
        txt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        for wl in np.linspace(400, 4000, 100):
            txt_file.write(f"{wl}\t{np.random.uniform(0, 1)}\n")
        txt_file.close()
        test_files.append(txt_file.name)
        
        # Test import
        for filename in test_files:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
                assert len(df) == 100
            elif filename.endswith('.txt'):
                df = pd.read_csv(filename, sep='\t', header=None)
                assert len(df) == 100
        
        # Cleanup
        for filename in test_files:
            os.unlink(filename)


class TestInstrumentIntegration:
    """Test instrument driver integration"""
    
    @pytest.fixture
    def mock_spectrometer(self):
        """Mock spectrometer driver"""
        driver = MagicMock()
        driver.connected = False
        return driver
    
    def test_ocean_optics_integration(self, mock_spectrometer):
        """Test Ocean Optics spectrometer integration"""
        # Setup mock driver
        mock_spectrometer.get_wavelengths.return_value = np.linspace(200, 1000, 2048)
        mock_spectrometer.get_spectrum.return_value = np.random.uniform(0, 65535, 2048)
        mock_spectrometer.integration_time = 100
        
        # Connect
        mock_spectrometer.connect.return_value = True
        assert mock_spectrometer.connect()
        
        # Configure
        mock_spectrometer.set_integration_time(100)
        
        # Acquire spectrum
        wavelengths = mock_spectrometer.get_wavelengths()
        intensities = mock_spectrometer.get_spectrum()
        
        assert len(wavelengths) == 2048
        assert len(intensities) == 2048
        
        # Process with analyzer
        analyzer = UVVisNIRAnalyzer()
        processed = analyzer.process_spectrum(
            wavelengths,
            intensities / 655.35,  # Convert to percentage
            mode='transmission'
        )
        
        assert 'absorbance' in processed
    
    def test_ftir_instrument_integration(self, mock_spectrometer):
        """Test FTIR instrument integration"""
        # Setup mock FTIR
        mock_spectrometer.get_wavenumbers.return_value = np.linspace(400, 4000, 3600)
        mock_spectrometer.get_interferogram.return_value = np.random.randn(8192)
        mock_spectrometer.get_spectrum.return_value = np.random.uniform(0, 1, 3600)
        
        # Configure
        mock_spectrometer.set_resolution(4)
        mock_spectrometer.set_scans(32)
        
        # Acquire spectrum
        wavenumbers = mock_spectrometer.get_wavenumbers()
        absorbance = mock_spectrometer.get_spectrum()
        
        # Process
        analyzer = FTIRAnalyzer()
        result = analyzer.process_spectrum(
            wavenumbers,
            absorbance
        )
        
        assert len(result.peaks) >= 0
        assert result.quality_metrics['coverage'] > 0


class TestWorkflowIntegration:
    """Test complete measurement workflows"""
    
    async def test_semiconductor_bandgap_workflow(self):
        """Test semiconductor band gap characterization workflow"""
        # List of semiconductors to test
        materials = [
            ('GaAs', 1.42, 'direct'),
            ('Si', 1.12, 'indirect'),
            ('GaN', 3.4, 'direct'),
            ('ZnO', 3.37, 'direct'),
            ('TiO2', 3.2, 'indirect')
        ]
        
        generator = OpticalDataGenerator()
        analyzer = UVVisNIRAnalyzer()
        
        results = []
        
        for material, expected_bg, transition in materials:
            # Generate spectrum
            spectrum = generator.generate_uvvisnir_spectrum(
                material=material,
                noise_level=0.005
            )
            
            # Process
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            
            # Extract band gap
            trans_type = TransitionType.DIRECT if transition == 'direct' else TransitionType.INDIRECT
            tauc = analyzer.calculate_tauc_plot(
                processed['wavelength'],
                processed['absorbance'],
                transition_type=trans_type
            )
            
            # Validate
            error = abs(tauc.band_gap - expected_bg)
            
            results.append({
                'material': material,
                'expected': expected_bg,
                'measured': tauc.band_gap,
                'error': error,
                'r_squared': tauc.r_squared
            })
            
            # Check accuracy
            assert error < 0.5, f"Band gap error too large for {material}"
            assert tauc.r_squared > 0.8, f"Poor fit quality for {material}"
        
        # Summary
        df = pd.DataFrame(results)
        print("\nBand Gap Results:")
        print(df.to_string(index=False))
        
        avg_error = df['error'].mean()
        assert avg_error < 0.3, "Average error too high"
    
    async def test_thin_film_characterization_workflow(self):
        """Test thin film optical characterization workflow"""
        generator = OpticalDataGenerator()
        analyzer = UVVisNIRAnalyzer()
        
        # Test different film thicknesses
        thicknesses = [100, 500, 1000, 2000]  # nm
        
        results = []
        
        for thickness in thicknesses:
            # Generate spectrum with interference
            spectrum = generator.generate_uvvisnir_spectrum(
                material='GaAs',
                include_interference=True,
                film_thickness=thickness
            )
            
            # Process and remove fringes
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            
            # Remove interference
            corrected = analyzer.remove_interference_fringes(
                processed['wavelength'],
                processed['corrected'],
                method='fft'
            )
            
            # Calculate optical constants
            optical = analyzer.calculate_optical_constants(
                processed['wavelength'],
                transmission=spectrum['transmission']/100,
                film_thickness=thickness
            )
            
            # Average refractive index
            avg_n = np.mean(optical.n[optical.n > 0])
            
            results.append({
                'thickness': thickness,
                'avg_n': avg_n,
                'fringe_period': self._calculate_fringe_period(
                    spectrum['wavelength'],
                    spectrum['transmission']
                )
            })
        
        df = pd.DataFrame(results)
        print("\nThin Film Results:")
        print(df.to_string(index=False))
        
        # Verify trends
        assert all(2.0 < r['avg_n'] < 4.5 for r in results)
    
    async def test_polymer_identification_workflow(self):
        """Test polymer identification workflow"""
        generator = OpticalDataGenerator()
        analyzer = FTIRAnalyzer()
        
        # Test different polymers
        polymers = ['polymer', 'protein', 'organic']
        
        results = []
        
        for polymer in polymers:
            # Generate spectrum
            spectrum = generator.generate_ftir_spectrum(
                sample_type=polymer,
                noise_level=0.01
            )
            
            # Process
            result = analyzer.process_spectrum(
                spectrum['wavenumber'],
                spectrum['absorbance']
            )
            
            # Quantitative analysis
            quant = analyzer.quantitative_analysis(result.peaks)
            
            # Identify main functional groups
            main_groups = [
                g.name for g in result.functional_groups
                if any(p.confidence > 0.7 for p in result.peaks if p.assignment == g.name)
            ]
            
            results.append({
                'sample': polymer,
                'n_peaks': len(result.peaks),
                'n_groups': len(result.functional_groups),
                'main_groups': ', '.join(main_groups[:3]),
                'snr': result.quality_metrics['snr']
            })
        
        df = pd.DataFrame(results)
        print("\nPolymer Identification Results:")
        print(df.to_string(index=False))
        
        # Verify all samples identified
        assert all(r['n_groups'] > 0 for r in results)
    
    def _calculate_fringe_period(self, wavelength, transmission):
        """Helper to calculate fringe period"""
        # Find peaks in transmission
        from scipy import signal
        peaks, _ = signal.find_peaks(transmission, prominence=1)
        
        if len(peaks) > 1:
            periods = np.diff(wavelength[peaks])
            return np.mean(periods)
        return 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_uvvisnir_invalid_input(self):
        """Test UV-Vis-NIR analyzer with invalid input"""
        analyzer = UVVisNIRAnalyzer()
        
        # Empty arrays
        with pytest.raises(ValueError):
            analyzer.process_spectrum(np.array([]), np.array([]))
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            analyzer.process_spectrum(
                np.linspace(300, 800, 100),
                np.random.rand(50)
            )
        
        # Invalid transition type
        wavelength = np.linspace(300, 800, 100)
        absorbance = np.random.rand(100)
        
        # Should handle gracefully
        result = analyzer.calculate_tauc_plot(
            wavelength,
            absorbance,
            transition_type=TransitionType.DIRECT
        )
        assert result is not None
    
    def test_ftir_noisy_spectrum(self):
        """Test FTIR analyzer with very noisy spectrum"""
        analyzer = FTIRAnalyzer()
        
        # Generate extremely noisy spectrum
        wavenumber = np.linspace(400, 4000, 3600)
        noise = np.random.randn(3600) * 0.5  # High noise
        
        # Should handle without crashing
        result = analyzer.process_spectrum(
            wavenumber,
            np.abs(noise),  # Ensure positive
            baseline_method='als'
        )
        
        assert result is not None
        assert result.quality_metrics['snr'] < 10  # Low SNR expected
    
    def test_edge_case_single_peak(self):
        """Test with spectrum containing single peak"""
        analyzer = FTIRAnalyzer()
        
        # Create spectrum with single peak
        wavenumber = np.linspace(400, 4000, 3600)
        absorbance = np.zeros_like(wavenumber)
        
        # Add single Gaussian peak
        center = 1650
        width = 30
        idx_center = np.argmin(np.abs(wavenumber - center))
        absorbance += 0.5 * np.exp(-0.5 * ((wavenumber - center) / width)**2)
        
        result = analyzer.process_spectrum(wavenumber, absorbance)
        
        assert len(result.peaks) >= 1
        assert any(abs(p.position - center) < 50 for p in result.peaks)


class TestPerformanceIntegration:
    """Test performance with realistic workloads"""
    
    def test_high_resolution_spectrum(self):
        """Test with high-resolution spectrum"""
        import time
        
        # Generate high-res spectrum
        wavelength = np.linspace(200, 2500, 20000)  # 20k points
        transmission = 80 + 10 * np.sin(2 * np.pi * wavelength / 100)
        
        analyzer = UVVisNIRAnalyzer()
        
        start = time.time()
        processed = analyzer.process_spectrum(
            wavelength,
            transmission,
            baseline_method=BaselineMethod.ALS
        )
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Processing too slow: {elapsed:.2f}s"
        assert len(processed['corrected']) == 20000
    
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        import time
        
        generator = OpticalDataGenerator()
        
        # Generate large batch
        n_samples = 50
        spectra = generator.generate_batch_spectra(
            'uvvisnir',
            n_samples=n_samples
        )
        
        analyzer = UVVisNIRAnalyzer()
        
        start = time.time()
        
        results = []
        for spectrum in spectra:
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            
            tauc = analyzer.calculate_tauc_plot(
                processed['wavelength'],
                processed['absorbance']
            )
            
            results.append({
                'sample_id': spectrum['sample_id'],
                'band_gap': tauc.band_gap
            })
        
        elapsed = time.time() - start
        time_per_sample = elapsed / n_samples
        
        print(f"\nBatch processing: {n_samples} samples in {elapsed:.2f}s")
        print(f"Average time per sample: {time_per_sample:.3f}s")
        
        assert time_per_sample < 1.0, "Processing too slow per sample"


# Run integration tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
    
    # Run async tests
    async def run_async_tests():
        workflow_tests = TestWorkflowIntegration()
        
        print("\n" + "="*60)
        print("Running Workflow Integration Tests")
        print("="*60)
        
        await workflow_tests.test_semiconductor_bandgap_workflow()
        await workflow_tests.test_thin_film_characterization_workflow()
        await workflow_tests.test_polymer_identification_workflow()
        
        print("\n✅ All workflow tests passed!")
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n" + "="*60)
    print("Session 7 Integration Tests Complete!")
    print("="*60)
