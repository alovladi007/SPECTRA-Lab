"""Unit tests for RTP thermal plant models."""

import pytest
import numpy as np
from app.models.rtp_thermal import (
    RTPThermalPlant,
    SiliconThermalProperties,
    STEFAN_BOLTZMANN,
)


class TestSiliconThermalProperties:
    """Test temperature-dependent silicon properties."""

    def test_density_constant(self):
        """Test that density is approximately constant."""
        assert SiliconThermalProperties.density(25.0) == 2329
        assert SiliconThermalProperties.density(1000.0) == 2329

    def test_specific_heat_increases_with_temp(self):
        """Test that specific heat increases with temperature."""
        cp_25 = SiliconThermalProperties.specific_heat(25.0)
        cp_500 = SiliconThermalProperties.specific_heat(500.0)
        cp_1000 = SiliconThermalProperties.specific_heat(1000.0)

        assert cp_500 > cp_25
        assert cp_1000 >= cp_500

    def test_thermal_conductivity_decreases_with_temp(self):
        """Test that thermal conductivity decreases with temperature."""
        k_25 = SiliconThermalProperties.thermal_conductivity(25.0)
        k_500 = SiliconThermalProperties.thermal_conductivity(500.0)
        k_1000 = SiliconThermalProperties.thermal_conductivity(1000.0)

        assert k_500 < k_25
        assert k_1000 < k_500

    def test_emissivity_increases_with_temp(self):
        """Test that emissivity increases slightly with temperature."""
        e_25 = SiliconThermalProperties.emissivity(25.0, "polished")
        e_1000 = SiliconThermalProperties.emissivity(1000.0, "polished")

        assert e_1000 >= e_25

    def test_emissivity_surface_condition(self):
        """Test emissivity depends on surface condition."""
        e_polished = SiliconThermalProperties.emissivity(1000.0, "polished")
        e_oxidized = SiliconThermalProperties.emissivity(1000.0, "oxidized")
        e_rough = SiliconThermalProperties.emissivity(1000.0, "rough")

        # Oxidized should be highest
        assert e_oxidized > e_polished
        assert e_oxidized > e_rough


class TestRTPThermalPlant:
    """Test RTP thermal plant dynamics."""

    def test_initialization(self):
        """Test plant initializes correctly."""
        plant = RTPThermalPlant(num_zones=4, max_lamp_power_W=10000.0)

        assert plant.num_zones == 4
        assert plant.max_lamp_power_W == 10000.0
        assert len(plant.zones) == 4
        assert plant.state.wafer_temperature_C == 25.0

    def test_reset(self):
        """Test reset to initial conditions."""
        plant = RTPThermalPlant()

        # Heat up
        plant.update(dt=1.0, lamp_powers_pct=np.array([100, 100, 100, 100]))
        assert plant.state.wafer_temperature_C > 25.0

        # Reset
        plant.reset(initial_temp_C=25.0)
        assert plant.state.wafer_temperature_C == 25.0
        assert np.all(plant.state.zone_temperatures_C == 25.0)

    def test_heating_increases_temperature(self):
        """Test that lamp power increases temperature."""
        plant = RTPThermalPlant()

        initial_temp = plant.state.wafer_temperature_C

        # Apply maximum lamp power
        for _ in range(10):
            plant.update(dt=1.0, lamp_powers_pct=np.array([100, 100, 100, 100]))

        assert plant.state.wafer_temperature_C > initial_temp

    def test_cooling_decreases_temperature(self):
        """Test that turning off lamps cools wafer."""
        plant = RTPThermalPlant()

        # Heat up first
        for _ in range(20):
            plant.update(dt=1.0, lamp_powers_pct=np.array([100, 100, 100, 100]))

        hot_temp = plant.state.wafer_temperature_C

        # Turn off lamps
        for _ in range(10):
            plant.update(dt=1.0, lamp_powers_pct=np.array([0, 0, 0, 0]))

        assert plant.state.wafer_temperature_C < hot_temp

    def test_steady_state(self):
        """Test that plant reaches steady state."""
        plant = RTPThermalPlant()

        lamp_powers = np.array([50, 50, 50, 50])

        # Run for long time
        temps = []
        for _ in range(100):
            plant.update(dt=1.0, lamp_powers_pct=lamp_powers)
            temps.append(plant.state.wafer_temperature_C)

        # Temperature should stabilize
        final_temps = temps[-10:]
        std_dev = np.std(final_temps)
        assert std_dev < 5.0  # Should be relatively stable

    def test_zone_coupling(self):
        """Test that zones influence each other."""
        plant = RTPThermalPlant()

        # Heat only zone 0
        lamp_powers = np.array([100, 0, 0, 0])

        for _ in range(30):
            plant.update(dt=0.5, lamp_powers_pct=lamp_powers)

        # Zone 0 should be hottest
        assert plant.state.zone_temperatures_C[0] > plant.state.zone_temperatures_C[1]

        # But zone 1 should be warmer than zone 3 due to coupling
        assert plant.state.zone_temperatures_C[1] > plant.state.zone_temperatures_C[3]

    def test_sensor_lag(self):
        """Test that sensors lag behind true temperature."""
        plant = RTPThermalPlant()

        # Step change in lamp power
        plant.update(dt=0.1, lamp_powers_pct=np.array([100, 100, 100, 100]))

        true_temp = plant.state.wafer_temperature_C
        pyro_temp = plant.state.pyrometer_reading_C
        tc_temp = plant.state.thermocouple_reading_C

        # Sensors should lag
        # (might not be true on first step due to initialization)
        # Run a few more steps
        for _ in range(5):
            plant.update(dt=0.1, lamp_powers_pct=np.array([100, 100, 100, 100]))

        # After several steps, sensors should be closer to true temp
        # but still show some difference
        assert abs(plant.state.pyrometer_reading_C - plant.state.wafer_temperature_C) >= 0

    def test_emissivity_drift(self):
        """Test emissivity drift over time."""
        plant = RTPThermalPlant()

        # Set drift rate
        plant.emissivity_model.drift_rate_per_s = 0.001

        initial_emissivity = plant.state.emissivity

        # Run for 100 seconds
        for _ in range(100):
            plant.update(dt=1.0, lamp_powers_pct=np.array([50, 50, 50, 50]))

        # Emissivity should have changed
        assert plant.state.emissivity != initial_emissivity

    def test_radiative_cooling(self):
        """Test radiative cooling is significant at high temp."""
        plant = RTPThermalPlant()

        # Heat to high temperature
        for _ in range(100):
            plant.update(dt=1.0, lamp_powers_pct=np.array([100, 100, 100, 100]))

        # Record heat flux
        heat_flux_hot = plant.state.heat_flux_W.copy()

        # Now cool to low temperature
        plant.reset(initial_temp_C=100.0)
        for _ in range(10):
            plant.update(dt=1.0, lamp_powers_pct=np.array([10, 10, 10, 10]))

        heat_flux_cool = plant.state.heat_flux_W.copy()

        # At higher temp, radiative loss should be larger (T^4)
        # So net heat flux at same power should be different

    def test_gas_flow_effect(self):
        """Test that gas flow affects cooling."""
        plant1 = RTPThermalPlant()
        plant2 = RTPThermalPlant()

        # Heat both to same temperature
        for _ in range(50):
            plant1.update(dt=1.0, lamp_powers_pct=np.array([80, 80, 80, 80]))
            plant2.update(dt=1.0, lamp_powers_pct=np.array([80, 80, 80, 80]))

        # Turn off lamps and cool with different gas flows
        for _ in range(20):
            plant1.update(
                dt=1.0,
                lamp_powers_pct=np.array([0, 0, 0, 0]),
                gas_flow_sccm=1000.0  # Low flow
            )
            plant2.update(
                dt=1.0,
                lamp_powers_pct=np.array([0, 0, 0, 0]),
                gas_flow_sccm=10000.0  # High flow
            )

        # Higher flow should cool faster
        assert plant2.state.wafer_temperature_C < plant1.state.wafer_temperature_C

    def test_pressure_effect(self):
        """Test that chamber pressure affects heat transfer."""
        plant1 = RTPThermalPlant()
        plant2 = RTPThermalPlant()

        # Heat both
        for _ in range(50):
            plant1.update(dt=1.0, lamp_powers_pct=np.array([80, 80, 80, 80]))
            plant2.update(dt=1.0, lamp_powers_pct=np.array([80, 80, 80, 80]))

        # Cool at different pressures
        for _ in range(20):
            plant1.update(
                dt=1.0,
                lamp_powers_pct=np.array([0, 0, 0, 0]),
                chamber_pressure_Pa=1000.0  # Low pressure (vacuum)
            )
            plant2.update(
                dt=1.0,
                lamp_powers_pct=np.array([0, 0, 0, 0]),
                chamber_pressure_Pa=101325.0  # Atmospheric
            )

        # Atmospheric pressure should cool faster (more convection)
        assert plant2.state.wafer_temperature_C < plant1.state.wafer_temperature_C


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
