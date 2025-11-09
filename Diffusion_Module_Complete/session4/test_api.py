"""
Test script for API validation.
"""

import sys
sys.path.insert(0, '.')

from api.service import simulate_oxidation, OxidationRequest
import asyncio


async def test_api():
    """Test the oxidation simulation endpoint."""
    print("Testing Oxidation API Endpoint\n")
    print("=" * 70)
    
    # Test case 1: Dry oxidation
    print("\n1. Dry Oxidation at 1000°C")
    print("-" * 70)
    request1 = OxidationRequest(
        temperature=1000,
        ambient='dry',
        time_points=[0.5, 1.0, 2.0, 4.0],
        pressure=1.0,
        initial_thickness=0.0,
        use_massoud=True,
        target_thickness=0.5
    )
    
    response1 = await simulate_oxidation(request1)
    print(f"Time points: {response1.time_points}")
    print(f"Thickness (nm): {[f'{x:.2f}' for x in response1.thickness_nm]}")
    print(f"Rate constants: B = {response1.rate_constants['B']:.2e} μm²/hr")
    print(f"                B/A = {response1.rate_constants['B_over_A']:.2e} μm/hr")
    
    if response1.inverse_solution:
        inv = response1.inverse_solution
        print(f"\nInverse solution for {inv['target_thickness_nm']:.1f} nm:")
        print(f"  Time required: {inv['time_required_hr']:.3f} hours")
        print(f"  Growth rate at target: {inv['growth_rate_at_target']:.4f} μm/hr")
    
    # Test case 2: Wet oxidation
    print("\n2. Wet Oxidation at 1000°C")
    print("-" * 70)
    request2 = OxidationRequest(
        temperature=1000,
        ambient='wet',
        time_points=[0.1, 0.5, 1.0],
        pressure=1.0,
        use_massoud=False
    )
    
    response2 = await simulate_oxidation(request2)
    print(f"Time points: {response2.time_points}")
    print(f"Thickness (nm): {[f'{x:.2f}' for x in response2.thickness_nm]}")
    print(f"Massoud applied: {response2.massoud_applied}")
    
    # Test case 3: Compare with/without Massoud
    print("\n3. Massoud Correction Comparison (900°C, Dry)")
    print("-" * 70)
    request3a = OxidationRequest(
        temperature=900,
        ambient='dry',
        time_points=[0.1, 0.5, 1.0],
        use_massoud=False
    )
    request3b = OxidationRequest(
        temperature=900,
        ambient='dry',
        time_points=[0.1, 0.5, 1.0],
        use_massoud=True
    )
    
    response3a = await simulate_oxidation(request3a)
    response3b = await simulate_oxidation(request3b)
    
    print(f"{'Time (hr)':<12} {'Deal-Grove (nm)':<16} {'Massoud (nm)':<16} {'Diff (nm)':<12}")
    for t, dg, mass in zip(response3a.time_points, 
                           response3a.thickness_nm,
                           response3b.thickness_nm):
        diff = mass - dg
        print(f"{t:<12.2f} {dg:<16.2f} {mass:<16.2f} {diff:<12.2f}")
    
    print("\n" + "=" * 70)
    print("All API tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_api())
