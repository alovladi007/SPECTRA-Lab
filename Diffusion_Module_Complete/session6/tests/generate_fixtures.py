"""
Generate synthetic test fixtures for Session 6 IO testing.

Creates:
- mes_diffusion_runs.csv
- fdc_furnace_data.parquet
- spc_charts.csv

Status: TEST UTILITY
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import random


def generate_mes_fixture(output_dir: Path):
    """Generate synthetic MES diffusion run data."""
    data = []

    # Generate 10 synthetic runs
    for i in range(10):
        run_id = f"RUN_{i+1:03d}"
        lot_id = f"LOT_{(i // 3) + 1:02d}"

        # Randomize process parameters
        processes = ["predeposition", "drive_in", "two_step"]
        process = random.choice(processes)

        dopants = ["B", "P", "As", "Sb"]
        dopant = random.choice(dopants)

        ambients = ["dry_O2", "wet_O2", "N2"]
        ambient = random.choice(ambients)

        # Temperature: 900-1100C
        temp = random.randint(900, 1100)

        # Time: 15-120 minutes
        time = random.randint(15, 120)

        # Start time
        start_time = datetime(2025, 11, 1) + timedelta(hours=i*2)
        end_time = start_time + timedelta(minutes=time)

        record = {
            'run_id': run_id,
            'lot_id': lot_id,
            'wafer_id': f"W{i+1:02d}",
            'process_type': process,
            'recipe_name': f"RECIPE_{dopant}_{process.upper()}",
            'equipment_id': f"FURNACE_0{(i % 3) + 1}",
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
            'temperature': temp,
            'temperature_unit': 'C',
            'time': time,
            'time_unit': 'min',
            'ambient': ambient,
            'pressure': round(random.uniform(0.9, 1.1), 3),
            'pressure_unit': 'atm',
            'flow_rate': random.randint(1000, 5000),
            'flow_rate_unit': 'sccm',
            'dopant_type': dopant,
            'target_concentration': f"{random.uniform(1e19, 1e20):.2e}",
            'concentration_unit': 'cm^-3',
            'target_junction_depth': random.randint(200, 800),
            'depth_unit': 'nm',
            'target_sheet_resistance': random.randint(50, 200),
            'status': 'completed',
            'measured_junction_depth': random.randint(200, 800),
            'measured_sheet_resistance': random.randint(50, 200),
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / "mes_diffusion_runs.csv", index=False)
    print(f"✓ Created {output_dir / 'mes_diffusion_runs.csv'}")


def generate_fdc_fixture(output_dir: Path):
    """Generate synthetic FDC furnace sensor data."""
    # Generate 1 hour of data at 1-second intervals
    start_time = datetime(2025, 11, 1, 10, 0, 0)

    data = []
    setpoint = 1000.0  # Target 1000C

    for i in range(3600):  # 1 hour
        timestamp = start_time + timedelta(seconds=i)

        # Simulate temperature with noise and ramp
        if i < 300:  # First 5 minutes: ramp up
            temp = 25 + (setpoint - 25) * (i / 300)
        else:  # Steady state with small variations
            temp = setpoint + random.gauss(0, 2)

        # Simulate pressure
        pressure = 760 + random.gauss(0, 5)

        # Simulate flow
        flow = 2000 + random.gauss(0, 50)

        # Occasional alarms
        temp_alarm = abs(temp - setpoint) > 10
        pressure_alarm = abs(pressure - 760) > 20

        record = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'temperature': round(temp, 2),
            'temperature_unit': 'C',
            'temperature_setpoint': setpoint,
            'pressure': round(pressure, 2),
            'pressure_unit': 'torr',
            'flow_rate': round(flow, 1),
            'flow_unit': 'sccm',
            'temp_alarm': temp_alarm,
            'pressure_alarm': pressure_alarm,
        }
        data.append(record)

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_dir / "fdc_furnace_data.parquet")
    print(f"✓ Created {output_dir / 'fdc_furnace_data.parquet'}")


def generate_spc_fixture(output_dir: Path):
    """Generate synthetic SPC control chart data."""
    # Generate junction depth measurements over 30 runs
    target = 500.0  # 500 nm target
    ucl = 550.0
    lcl = 450.0
    usl = 560.0
    lsl = 440.0

    data = []
    start_time = datetime(2025, 11, 1)

    for i in range(30):
        timestamp = start_time + timedelta(hours=i*2)
        run_id = f"RUN_{i+1:03d}"

        # Simulate measurements with occasional out-of-control points
        if i in [10, 25]:  # Introduce some OOC points
            value = random.choice([usl + 5, lsl - 5])
            out_of_control = True
            violation_rules = ['Rule 1: Beyond control limits']
        elif i in [15, 16, 17, 18]:  # Trend violation
            value = target + (i - 15) * 10
            out_of_control = True
            violation_rules = ['Rule 6: Trend of 4 points']
        else:
            value = random.gauss(target, 15)
            out_of_control = False
            violation_rules = []

        record = {
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'run_id': run_id,
            'value': round(value, 2),
            'subgroup_id': f"SG_{(i // 5) + 1}",
            'sample_size': 1,
            'out_of_control': out_of_control,
            'violation_rules': ', '.join(violation_rules) if violation_rules else '',
            'ucl': ucl,
            'lcl': lcl,
            'usl': usl,
            'lsl': lsl,
            'target': target,
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(output_dir / "spc_charts.csv", index=False)
    print(f"✓ Created {output_dir / 'spc_charts.csv'}")


if __name__ == "__main__":
    # Create fixtures directory
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    print("Generating test fixtures...")
    generate_mes_fixture(fixtures_dir)
    generate_fdc_fixture(fixtures_dir)
    generate_spc_fixture(fixtures_dir)
    print("\n✓ All fixtures generated successfully!")
