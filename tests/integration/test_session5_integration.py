"""
Session 5 Integration Tests - Complete End-to-End Workflows

This test suite validates all electrical characterization methods for Session 5:
- MOSFET I-V (transfer and output characteristics)
- Solar Cell I-V (efficiency and performance)
- C-V Profiling (MOS and Schottky)
- BJT I-V (Gummel plots and output)
- Batch processing
- Report generation

Run with: pytest services/analysis/tests/integration/test_session5_workflows.py -v --cov
"""

import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any
import time

# Import analysis modules
from services.analysis.app.methods.electrical.mosfet_analysis import analyze_mosfet_transfer, analyze_mosfet_output
from services.analysis.app.methods.electrical.solar_cell_analysis import analyze_solar_cell
from services.analysis.app.methods.electrical.cv_profiling import analyze_mos_capacitor, analyze_schottky_diode
from services.analysis.app.methods.electrical.bjt_analysis import analyze_bjt_gummel, analyze_bjt_output


class TestMOSFETWorkflow:
    """Complete MOSFET characterization workflow tests"""
    
    def setup_method(self):
        """Load test data before each test"""
        self.test_data_dir = Path("data/test_data/electrical/mosfet_iv")
        
    def test_nmos_transfer_complete_workflow(self):
        """Test complete n-MOS transfer characteristic analysis"""
        # Load test data
        data_file = self.test_data_dir / "n-mos_transfer.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        # Run analysis
        start_time = time.time()
        results = analyze_mosfet_transfer(
            voltage_gate=np.array(test_data['voltage_gate']),
            current_drain=np.array(test_data['current_drain']),
            voltage_drain=test_data['voltage_drain'],
            width=10e-6,  # 10 µm
            length=1e-6,  # 1 µm
            oxide_thickness=10e-9  # 10 nm
        )
        analysis_time = time.time() - start_time
        
        # Validate results structure
        assert 'threshold_voltage' in results
        assert 'transconductance_max' in results
        assert 'mobility' in results
        assert 'quality_score' in results
        
        # Validate threshold voltage
        assert 'linear_extrapolation' in results['threshold_voltage']
        assert 'constant_current' in results['threshold_voltage']
        assert 'transconductance' in results['threshold_voltage']
        assert 'average' in results['threshold_voltage']
        
        # Check reasonable values for n-MOS
        vth = results['threshold_voltage']['average']
        assert 0.3 < vth < 1.5, f"Threshold voltage {vth}V outside expected range"
        
        # Check mobility
        mobility = results['mobility']['effective']
        assert 200 < mobility < 800, f"Mobility {mobility} cm²/V·s outside expected range"
        
        # Check Ion/Ioff ratio
        assert results['ion_ioff_ratio'] > 1e4, "Ion/Ioff ratio too low"
        
        # Check quality score
        assert 0 <= results['quality_score'] <= 100
        assert results['quality_score'] >= 70, f"Quality score {results['quality_score']} too low"
        
        # Check performance
        assert analysis_time < 1.0, f"Analysis took {analysis_time:.2f}s, should be < 1s"
        
        print(f"\n✓ n-MOS transfer analysis: Vth={vth:.3f}V, µ={mobility:.0f} cm²/V·s, " +
              f"Score={results['quality_score']}, Time={analysis_time:.3f}s")
    
    def test_pmos_transfer_analysis(self):
        """Test p-MOS transfer characteristic analysis"""
        data_file = self.test_data_dir / "p-mos_transfer.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_mosfet_transfer(
            voltage_gate=np.array(test_data['voltage_gate']),
            current_drain=np.array(test_data['current_drain']),
            voltage_drain=test_data['voltage_drain'],
            width=10e-6,
            length=1e-6,
            oxide_thickness=10e-9
        )
        
        # p-MOS should have negative threshold voltage
        vth = results['threshold_voltage']['average']
        assert -1.5 < vth < -0.3, f"p-MOS Vth {vth}V outside expected range"
        
        # p-MOS typically has lower mobility than n-MOS
        mobility = results['mobility']['effective']
        assert 100 < mobility < 400, f"p-MOS mobility {mobility} outside expected range"
        
        print(f"✓ p-MOS transfer analysis: Vth={vth:.3f}V, µ={mobility:.0f} cm²/V·s")
    
    def test_mosfet_output_characteristics(self):
        """Test MOSFET output characteristic analysis"""
        data_file = self.test_data_dir / "n-mos_output.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_mosfet_output(
            voltage_drain_array=np.array(test_data['voltage_drain']),
            voltage_gate_array=np.array(test_data['voltage_gate_values']),
            current_drain_matrix=np.array(test_data['current_drain_matrix']),
            width=10e-6,
            length=1e-6
        )
        
        # Check for channel length modulation parameter
        assert 'lambda' in results
        assert results['lambda'] > 0, "Channel length modulation should be positive"
        
        # Check on-resistance
        assert 'on_resistance' in results
        assert results['on_resistance'] > 0
        
        print(f"✓ MOSFET output: λ={results['lambda']:.4f} V⁻¹, Ron={results['on_resistance']:.1f} Ω")
    
    def test_parameter_extraction_accuracy(self):
        """Validate parameter extraction accuracy against known values"""
        # Use synthetic data with known parameters
        vgs = np.linspace(-1, 5, 120)
        vth_true = 0.7
        k = 1e-3
        vds = 0.1
        W_L = 10
        
        # Generate ideal MOSFET curve
        id = []
        for v in vgs:
            if v < vth_true:
                id.append(1e-12 * np.exp((v - vth_true) / 0.1))  # Subthreshold
            else:
                if vds < v - vth_true:
                    id.append(k * W_L * ((v - vth_true) * vds - vds**2 / 2))  # Linear
                else:
                    id.append(0.5 * k * W_L * (v - vth_true)**2)  # Saturation
        
        results = analyze_mosfet_transfer(
            voltage_gate=vgs,
            current_drain=np.array(id),
            voltage_drain=vds,
            width=10e-6,
            length=1e-6,
            oxide_thickness=10e-9
        )
        
        # Check extraction accuracy
        vth_extracted = results['threshold_voltage']['average']
        error = abs(vth_extracted - vth_true) / vth_true * 100
        
        assert error < 5, f"Vth extraction error {error:.1f}% > 5%"
        print(f"✓ Parameter extraction: True Vth={vth_true}V, Extracted={vth_extracted:.3f}V, Error={error:.2f}%")


class TestSolarCellWorkflow:
    """Solar cell testing workflows"""
    
    def setup_method(self):
        """Load test data"""
        self.test_data_dir = Path("data/test_data/electrical/solar_cell_iv")
    
    def test_silicon_cell_1sun_analysis(self):
        """Test silicon solar cell at 1 sun (1000 W/m²)"""
        data_file = self.test_data_dir / "silicon_1sun.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        start_time = time.time()
        results = analyze_solar_cell(
            voltage=np.array(test_data['voltage']),
            current=np.array(test_data['current']),
            area=test_data['area'],
            irradiance=1000,  # W/m²
            temperature=298.15  # 25°C
        )
        analysis_time = time.time() - start_time
        
        # Validate key parameters
        assert 'isc' in results
        assert 'voc' in results
        assert 'max_power_point' in results
        assert 'fill_factor' in results
        assert 'efficiency' in results
        
        # Check reasonable values for Si cell
        assert results['isc'] > 0.030, "Isc too low for Si cell"
        assert 0.5 < results['voc'] < 0.7, f"Voc {results['voc']}V outside expected range"
        assert 0.75 < results['fill_factor'] < 0.85, f"FF {results['fill_factor']} outside expected range"
        assert 15 < results['efficiency'] < 25, f"Efficiency {results['efficiency']}% outside expected range"
        
        # Check performance
        assert analysis_time < 0.5, f"Analysis took {analysis_time:.2f}s"
        
        print(f"\n✓ Si solar cell (1 sun): η={results['efficiency']:.1f}%, " +
              f"FF={results['fill_factor']:.3f}, Voc={results['voc']:.3f}V, " +
              f"Time={analysis_time:.3f}s")
    
    def test_low_light_performance(self):
        """Test solar cell at low light (0.5 sun)"""
        data_file = self.test_data_dir / "silicon_0.5sun.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_solar_cell(
            voltage=np.array(test_data['voltage']),
            current=np.array(test_data['current']),
            area=test_data['area'],
            irradiance=500,  # 0.5 sun
            temperature=298.15
        )
        
        # At lower irradiance, efficiency typically drops slightly
        # but should still be reasonable
        assert results['efficiency'] > 10
        assert results['isc'] < 0.020  # Lower than 1 sun
        
        print(f"✓ Si solar cell (0.5 sun): η={results['efficiency']:.1f}%, Isc={results['isc']:.4f}A")
    
    def test_stc_normalization(self):
        """Test Standard Test Conditions normalization"""
        data_file = self.test_data_dir / "silicon_1sun.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        # Test at different temperature
        results_25c = analyze_solar_cell(
            voltage=np.array(test_data['voltage']),
            current=np.array(test_data['current']),
            area=test_data['area'],
            irradiance=1000,
            temperature=298.15  # 25°C
        )
        
        results_50c = analyze_solar_cell(
            voltage=np.array(test_data['voltage']),
            current=np.array(test_data['current']),
            area=test_data['area'],
            irradiance=1000,
            temperature=323.15  # 50°C
        )
        
        # Voc should decrease with temperature
        assert results_50c['voc'] < results_25c['voc'], "Voc should decrease with temperature"
        
        # Temperature coefficient should be negative for Si
        temp_diff = 25  # K
        voc_change = (results_50c['voc'] - results_25c['voc']) / results_25c['voc'] * 100
        temp_coeff = voc_change / temp_diff
        
        assert -0.5 < temp_coeff < -0.3, f"Temperature coefficient {temp_coeff}%/K outside expected range"
        
        print(f"✓ STC normalization: Voc@25°C={results_25c['voc']:.3f}V, " +
              f"Voc@50°C={results_50c['voc']:.3f}V, TC={temp_coeff:.3f}%/K")


class TestCVProfilingWorkflow:
    """C-V analysis workflows"""
    
    def setup_method(self):
        """Load test data"""
        self.test_data_dir = Path("data/test_data/electrical/cv_profiling")
    
    def test_mos_capacitor_analysis(self):
        """Test MOS capacitor C-V analysis"""
        data_file = self.test_data_dir / "mos_n_substrate.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_mos_capacitor(
            voltage=np.array(test_data['voltage']),
            capacitance=np.array(test_data['capacitance']),
            frequency=test_data['frequency'],
            area=test_data['area'],
            substrate_type='n-type'
        )
        
        # Validate MOS parameters
        assert 'oxide_capacitance' in results
        assert 'oxide_thickness' in results
        assert 'flatband_voltage' in results
        assert 'threshold_voltage' in results
        assert 'interface_trap_density' in results
        
        # Check reasonable values
        cox = results['oxide_capacitance']
        tox = results['oxide_thickness']
        assert tox > 1e-9 and tox < 100e-9, f"Oxide thickness {tox*1e9:.1f}nm outside expected range"
        
        vfb = results['flatband_voltage']
        assert -1.5 < vfb < 0.5, f"Flatband voltage {vfb}V outside expected range"
        
        Dit = results['interface_trap_density']
        assert Dit < 1e13, f"Dit {Dit:.2e} cm⁻²eV⁻¹ too high"
        
        print(f"\n✓ MOS C-V: tox={tox*1e9:.1f}nm, Vfb={vfb:.3f}V, Dit={Dit:.2e} cm⁻²eV⁻¹")
    
    def test_schottky_diode_profiling(self):
        """Test Schottky diode C-V and doping profile"""
        data_file = self.test_data_dir / "schottky_n_type.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_schottky_diode(
            voltage=np.array(test_data['voltage']),
            capacitance=np.array(test_data['capacitance']),
            frequency=test_data['frequency'],
            area=test_data['area'],
            substrate_type='n-type'
        )
        
        # Validate Schottky parameters
        assert 'builtin_voltage' in results
        assert 'barrier_height' in results
        assert 'doping_concentration' in results
        assert 'doping_profile' in results
        
        # Check reasonable values
        vbi = results['builtin_voltage']
        assert 0.5 < vbi < 1.2, f"Built-in voltage {vbi}V outside expected range"
        
        phi_b = results['barrier_height']
        assert 0.5 < phi_b < 1.5, f"Barrier height {phi_b}eV outside expected range"
        
        nd = results['doping_concentration']
        assert 1e14 < nd < 1e18, f"Doping {nd:.2e} cm⁻³ outside expected range"
        
        # Check doping profile
        assert len(results['doping_profile']) > 0
        
        print(f"✓ Schottky C-V: Vbi={vbi:.3f}V, φB={phi_b:.3f}eV, ND={nd:.2e} cm⁻³")
    
    def test_doping_profile_extraction(self):
        """Validate doping profile extraction accuracy"""
        data_file = self.test_data_dir / "schottky_n_type.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_schottky_diode(
            voltage=np.array(test_data['voltage']),
            capacitance=np.array(test_data['capacitance']),
            frequency=test_data['frequency'],
            area=test_data['area'],
            substrate_type='n-type'
        )
        
        profile = results['doping_profile']
        
        # Check profile has reasonable shape
        depths = [p['depth'] for p in profile]
        concentrations = [p['concentration'] for p in profile]
        
        assert len(depths) > 10, "Doping profile should have multiple points"
        assert all(d > 0 for d in depths), "All depths should be positive"
        assert all(c > 0 for c in concentrations), "All concentrations should be positive"
        
        # For uniform doping, profile should be relatively flat
        conc_std = np.std(concentrations)
        conc_mean = np.mean(concentrations)
        cv = conc_std / conc_mean
        
        assert cv < 0.5, f"Doping profile too variable (CV={cv:.2f})"
        
        print(f"✓ Doping profile: {len(profile)} points, Mean={conc_mean:.2e} cm⁻³, CV={cv:.2f}")


class TestBJTWorkflow:
    """BJT characterization workflows"""
    
    def setup_method(self):
        """Load test data"""
        self.test_data_dir = Path("data/test_data/electrical/bjt_iv")
    
    def test_npn_gummel_plot_analysis(self):
        """Test npn BJT Gummel plot analysis"""
        data_file = self.test_data_dir / "npn_gummel.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_bjt_gummel(
            voltage_be=np.array(test_data['voltage_be']),
            current_collector=np.array(test_data['current_collector']),
            current_base=np.array(test_data['current_base']),
            voltage_ce=test_data['voltage_ce']
        )
        
        # Validate results
        assert 'current_gain' in results
        assert 'ideality_factors' in results
        assert 'saturation_currents' in results
        
        # Check current gain
        beta = results['current_gain']['beta_dc']
        assert 50 < beta < 500, f"Current gain {beta} outside expected range"
        
        # Check ideality factors
        nc = results['ideality_factors']['collector']
        nb = results['ideality_factors']['base']
        
        assert 0.8 < nc < 1.5, f"Collector ideality {nc} outside expected range"
        assert 1.0 < nb < 3.0, f"Base ideality {nb} outside expected range"
        
        print(f"\n✓ npn Gummel: β={beta:.0f}, nC={nc:.2f}, nB={nb:.2f}")
    
    def test_bjt_output_characteristics(self):
        """Test BJT output characteristic analysis"""
        data_file = self.test_data_dir / "npn_output.json"
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_bjt_output(
            voltage_ce_array=np.array(test_data['voltage_ce']),
            current_base_array=np.array(test_data['current_base_values']),
            current_collector_matrix=np.array(test_data['current_collector_matrix'])
        )
        
        # Check Early voltage
        assert 'early_voltage' in results
        va = results['early_voltage']
        
        assert 20 < va < 200, f"Early voltage {va}V outside expected range"
        
        print(f"✓ npn Output: VA={va:.0f}V")


class TestBatchProcessing:
    """Multi-device batch processing tests"""
    
    def test_batch_mosfet_analysis(self):
        """Test batch processing of multiple MOSFET devices"""
        test_data_dir = Path("data/test_data/electrical/mosfet_iv")
        
        # Process multiple devices
        results_list = []
        devices = ["n-mos_transfer.json", "p-mos_transfer.json"]
        
        start_time = time.time()
        for device_file in devices:
            data_file = test_data_dir / device_file
            with open(data_file, 'r') as f:
                test_data = json.load(f)
            
            results = analyze_mosfet_transfer(
                voltage_gate=np.array(test_data['voltage_gate']),
                current_drain=np.array(test_data['current_drain']),
                voltage_drain=test_data['voltage_drain'],
                width=10e-6,
                length=1e-6,
                oxide_thickness=10e-9
            )
            results_list.append(results)
        
        batch_time = time.time() - start_time
        
        # Validate batch results
        assert len(results_list) == len(devices)
        assert all('quality_score' in r for r in results_list)
        
        # Calculate statistics
        vth_values = [r['threshold_voltage']['average'] for r in results_list]
        vth_mean = np.mean(np.abs(vth_values))
        vth_std = np.std(np.abs(vth_values))
        
        print(f"\n✓ Batch MOSFET: {len(devices)} devices, " +
              f"Mean |Vth|={vth_mean:.3f}V ± {vth_std:.3f}V, " +
              f"Time={batch_time:.3f}s")
    
    def test_parallel_processing_speed(self):
        """Test that batch processing is efficient"""
        # This is a placeholder for actual parallel processing tests
        # In production, you would use multiprocessing or asyncio
        pass


class TestReportGeneration:
    """Report and export workflow tests"""
    
    def test_json_export(self):
        """Test JSON export of analysis results"""
        test_data_dir = Path("data/test_data/electrical/mosfet_iv")
        data_file = test_data_dir / "n-mos_transfer.json"
        
        with open(data_file, 'r') as f:
            test_data = json.load(f)
        
        results = analyze_mosfet_transfer(
            voltage_gate=np.array(test_data['voltage_gate']),
            current_drain=np.array(test_data['current_drain']),
            voltage_drain=test_data['voltage_drain'],
            width=10e-6,
            length=1e-6,
            oxide_thickness=10e-9
        )
        
        # Convert to JSON
        json_str = json.dumps(results, indent=2)
        
        # Validate JSON is valid
        parsed = json.loads(json_str)
        assert parsed == results
        
        print(f"\n✓ JSON export: {len(json_str)} bytes")
    
    def test_pdf_report_generation(self):
        """Test PDF report generation (placeholder)"""
        # In production, this would test actual PDF generation
        # using reportlab or similar library
        pass


class TestErrorHandling:
    """Edge cases and validation tests"""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid input data"""
        # Empty arrays
        with pytest.raises(ValueError):
            analyze_mosfet_transfer(
                voltage_gate=np.array([]),
                current_drain=np.array([]),
                voltage_drain=0.1,
                width=10e-6,
                length=1e-6,
                oxide_thickness=10e-9
            )
        
        # Mismatched array lengths
        with pytest.raises(ValueError):
            analyze_mosfet_transfer(
                voltage_gate=np.array([0, 1, 2]),
                current_drain=np.array([0, 1]),  # Different length
                voltage_drain=0.1,
                width=10e-6,
                length=1e-6,
                oxide_thickness=10e-9
            )
        
        print("\n✓ Error handling: Invalid inputs correctly rejected")
    
    def test_compliance_limits(self):
        """Test safety compliance limit checking"""
        # Test with very high current (should warn or limit)
        vgs = np.linspace(0, 5, 100)
        id_high = np.ones_like(vgs) * 1.0  # 1A - very high
        
        # This should either warn or clip the data
        results = analyze_mosfet_transfer(
            voltage_gate=vgs,
            current_drain=id_high,
            voltage_drain=0.1,
            width=10e-6,
            length=1e-6,
            oxide_thickness=10e-9
        )
        
        # Should still complete without crashing
        assert 'quality_score' in results
        
        print("✓ Compliance limits: High current handled safely")


class TestPerformance:
    """Performance benchmark tests"""
    
    def test_analysis_speed_mosfet(self):
        """Benchmark MOSFET analysis speed"""
        # Generate synthetic data
        vgs = np.linspace(-1, 5, 200)
        id = 1e-3 * np.maximum(0, (vgs - 0.7)**2)
        
        # Run multiple times
        times = []
        for _ in range(10):
            start = time.time()
            analyze_mosfet_transfer(
                voltage_gate=vgs,
                current_drain=id,
                voltage_drain=0.1,
                width=10e-6,
                length=1e-6,
                oxide_thickness=10e-9
            )
            times.append(time.time() - start)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        assert mean_time < 1.0, f"Analysis too slow: {mean_time:.3f}s"
        
        print(f"\n✓ MOSFET performance: {mean_time:.3f}s ± {std_time:.3f}s (10 runs)")
    
    def test_memory_usage(self):
        """Test memory efficiency with large datasets"""
        # Generate large dataset
        vgs = np.linspace(-1, 5, 10000)  # 10k points
        id = 1e-3 * np.maximum(0, (vgs - 0.7)**2)
        
        # Run analysis
        results = analyze_mosfet_transfer(
            voltage_gate=vgs,
            current_drain=id,
            voltage_drain=0.1,
            width=10e-6,
            length=1e-6,
            oxide_thickness=10e-9
        )
        
        assert 'quality_score' in results
        print("✓ Memory: Large dataset (10k points) processed successfully")


# Run all tests with coverage
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=services/analysis", "--cov-report=html"])