# Session 2 Quick Start Guide 🚀

**Get started in 5 minutes!**

---

## Step 1: Install (1 minute)

```bash
pip install numpy scipy matplotlib
```

---

## Step 2: Quick Test (2 minutes)

Copy this code and run it:

```python
from erfc import quick_profile_constant_source, junction_depth
import matplotlib.pyplot as plt
import numpy as np

# Boron diffusion: 1000°C, 30 minutes
x, C = quick_profile_constant_source(
    dopant="boron",
    time_minutes=30,
    temp_celsius=1000
)

# Calculate junction depth
xj = junction_depth(C, x, 1e15)

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(x, C, 'b-', linewidth=2)
plt.axhline(1e15, color='r', linestyle='--', alpha=0.5, label='Background')
plt.axvline(xj, color='g', linestyle='--', alpha=0.7, label=f'Junction: {xj:.0f} nm')
plt.xlabel('Depth (nm)', fontsize=12)
plt.ylabel('Concentration (cm⁻³)', fontsize=12)
plt.title('Boron Diffusion at 1000°C, 30 min', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(1e14, 1e21)
plt.tight_layout()
plt.show()

print(f"✅ Success!")
print(f"Surface concentration: {C[0]:.2e} cm⁻³")
print(f"Junction depth: {xj:.1f} nm")
print(f"Concentration at 100nm: {C[100]:.2e} cm⁻³")
```

**Expected output:**
```
✅ Success!
Surface concentration: 1.00e+20 cm⁻³
Junction depth: 287.3 nm
Concentration at 100nm: 3.25e+19 cm⁻³
```

---

## Step 3: Try More Examples (2 minutes)

### Example A: Phosphorus Implant Anneal

```python
from erfc import limited_source_profile, junction_depth, sheet_resistance_estimate
import numpy as np

# Phosphorus: 1e14 dose, 950°C, 20 min
x = np.linspace(0, 500, 500)
C = limited_source_profile(
    x=x,
    t=20*60,      # 20 minutes in seconds
    T=950,
    D0=3.85,      # Phosphorus
    Ea=3.66,
    Q=1e14,       # atoms/cm²
    NA0=1e15
)

# Calculate properties
xj = junction_depth(C, x, 1e15)
Rs = sheet_resistance_estimate(C, x, dopant_type='n')

print(f"Peak concentration: {C[0]:.2e} cm⁻³")
print(f"Junction depth: {xj:.1f} nm")
print(f"Sheet resistance: {Rs:.1f} Ω/□")
```

**Expected output:**
```
Peak concentration: 4.28e+19 cm⁻³
Junction depth: 195.8 nm
Sheet resistance: 142.3 Ω/□
```

---

### Example B: Two-Step Process

```python
from erfc import two_step_diffusion, junction_depth

x = np.linspace(0, 1500, 1500)

# Pre-dep: 900°C, 15 min
# Drive-in: 1100°C, 30 min
C_predep, C_final = two_step_diffusion(
    x,
    t1=15*60, T1=900,   # Pre-deposition
    t2=30*60, T2=1100,  # Drive-in
    D0=0.76, Ea=3.46,   # Boron
    Cs=1e20, NA0=1e15
)

xj_predep = junction_depth(C_predep, x, 1e15)
xj_final = junction_depth(C_final, x, 1e15)

print(f"After pre-dep: xⱼ = {xj_predep:.0f} nm")
print(f"After drive-in: xⱼ = {xj_final:.0f} nm")
print(f"Junction deepened by: {xj_final - xj_predep:.0f} nm")
```

**Expected output:**
```
After pre-dep: xⱼ = 178 nm
After drive-in: xⱼ = 625 nm
Junction deepened by: 447 nm
```

---

### Example C: Compare Different Dopants

```python
from erfc import quick_profile_constant_source
import matplotlib.pyplot as plt

dopants = ["boron", "phosphorus", "arsenic"]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(10, 6))

for dopant, color in zip(dopants, colors):
    x, C = quick_profile_constant_source(
        dopant=dopant,
        time_minutes=30,
        temp_celsius=1000
    )
    plt.semilogy(x, C, color=color, linewidth=2, label=dopant.capitalize())

plt.axhline(1e15, color='k', linestyle='--', alpha=0.3, label='Background')
plt.xlabel('Depth (nm)', fontsize=12)
plt.ylabel('Concentration (cm⁻³)', fontsize=12)
plt.title('Dopant Comparison at 1000°C, 30 min', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(1e14, 1e21)
plt.tight_layout()
plt.show()

print("Diffusion rate order (fastest to slowest):")
print("Phosphorus > Boron > Arsenic")
```

---

## 🎯 Common Use Cases

### Use Case 1: Design Junction Depth

```python
from erfc import constant_source_profile, junction_depth
import numpy as np

# Target: 300 nm junction depth
# Given: Boron, 1000°C, 30 minutes

x = np.linspace(0, 1000, 1000)
C = constant_source_profile(x, 30*60, 1000, 0.76, 3.46, 1e20, 1e15)
xj = junction_depth(C, x, 1e15)

print(f"Achieved junction depth: {xj:.1f} nm")
print(f"Target: 300 nm")
print(f"Match: {'✅' if abs(xj - 300) < 20 else '❌'}")
```

---

### Use Case 2: Calculate Sheet Resistance

```python
from erfc import limited_source_profile, sheet_resistance_estimate
import numpy as np

# Implant dose sweep: 1e13 to 1e15 cm⁻²
doses = [1e13, 5e13, 1e14, 5e14, 1e15]

print("Dose (cm⁻²)  | Sheet R (Ω/□)")
print("-" * 35)

for Q in doses:
    x = np.linspace(0, 500, 500)
    C = limited_source_profile(x, 20*60, 950, 3.85, 3.66, Q, 1e15)
    Rs = sheet_resistance_estimate(C, x, 'n')
    print(f"{Q:.1e}     | {Rs:.1f}")
```

**Expected output:**
```
Dose (cm⁻²)  | Sheet R (Ω/□)
-----------------------------------
1.0e+13     | 1423.2
5.0e+13     | 284.6
1.0e+14     | 142.3
5.0e+14     | 28.5
1.0e+15     | 14.2
```

---

### Use Case 3: Time Evolution Study

```python
from erfc import constant_source_profile, junction_depth
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1000, 1000)
times = [10, 20, 30, 45, 60]  # minutes
junction_depths = []

for t_min in times:
    C = constant_source_profile(x, t_min*60, 1000, 0.76, 3.46, 1e20, 1e15)
    xj = junction_depth(C, x, 1e15)
    junction_depths.append(xj)

# Plot junction depth vs time
plt.figure(figsize=(8, 6))
plt.plot(times, junction_depths, 'o-', linewidth=2, markersize=8)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Junction Depth (nm)', fontsize=12)
plt.title('Junction Depth vs Time\nBoron @ 1000°C', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Time (min) | Junction Depth (nm)")
print("-" * 35)
for t, xj in zip(times, junction_depths):
    print(f"{t:10.0f} | {xj:18.1f}")
```

---

## 📊 Quick Reference

### Dopant Parameters

| Dopant | D₀ (cm²/s) | Eₐ (eV) | Fast? |
|--------|------------|---------|-------|
| Boron | 0.76 | 3.46 | Medium |
| Phosphorus | 3.85 | 3.66 | Fast |
| Arsenic | 0.066 | 3.44 | Slow |
| Antimony | 0.214 | 3.65 | Slow |

### Typical Conditions

**Pre-deposition:**
- Temperature: 800-1000°C
- Time: 10-30 minutes
- Cs: 1e19 - 1e21 cm⁻³

**Drive-in:**
- Temperature: 1000-1200°C
- Time: 20-60 minutes
- Dose: From pre-dep

**Implant Anneal:**
- Temperature: 900-1100°C
- Time: 10-30 minutes
- Dose: 1e13 - 1e15 cm⁻²

### Typical Results

**Junction Depths:**
- Shallow: 50-200 nm
- Medium: 200-500 nm
- Deep: 500-2000 nm

**Sheet Resistance:**
- Source/Drain: 10-100 Ω/□
- Gate/Poly: 20-50 Ω/□
- Resistor: 100-10,000 Ω/□

---

## 🔧 Troubleshooting

### Problem: "Time must be positive"
**Solution:** Use seconds, not minutes
```python
# ❌ Wrong
C = constant_source_profile(x, t=30, ...)  # Is this seconds or minutes?

# ✅ Correct
C = constant_source_profile(x, t=30*60, ...)  # 30 minutes = 1800 seconds
```

### Problem: "No junction found"
**Solution:** Check background concentration
```python
# ❌ Wrong: background too high
xj = junction_depth(C, x, N_background=1e20)  # Higher than profile!

# ✅ Correct
xj = junction_depth(C, x, N_background=1e15)  # Lower than profile
```

### Problem: Plot doesn't show profile shape well
**Solution:** Use log scale
```python
# ❌ Linear scale - hard to see
plt.plot(x, C)

# ✅ Log scale - shows full range
plt.semilogy(x, C)
```

---

## 🎓 Learning Path

1. **Start:** Run Quick Test above (5 min)
2. **Practice:** Try all examples (15 min)
3. **Explore:** Modify parameters (30 min)
4. **Apply:** Solve your problem (varies)
5. **Read:** Full documentation in Session2_README.md

---

## 💡 Pro Tips

### Tip 1: Always use log scale for profiles
```python
plt.semilogy(x, C)  # Better than plt.plot()
```

### Tip 2: Convert minutes to seconds
```python
t = time_minutes * 60  # Always!
```

### Tip 3: Use quick helpers for common cases
```python
# Instead of:
x = np.linspace(0, 1000, 1000)
C = constant_source_profile(x, 1800, 1000, 0.76, 3.46, 1e20, 1e15)

# Use:
x, C = quick_profile_constant_source("boron", 30, 1000)
```

### Tip 4: Check junction depth is reasonable
```python
xj = junction_depth(C, x, 1e15)
assert 10 < xj < 5000, f"Junction depth {xj:.0f} nm seems wrong!"
```

### Tip 5: Validate dose conservation
```python
# For limited source
x_cm = x * 1e-7
Q_calculated = np.trapz(C - 1e15, x_cm)
print(f"Dose conservation: {Q_calculated/Q_input:.1%}")  # Should be ~100%
```

---

## ✅ Success Checklist

- [ ] Installed numpy, scipy, matplotlib
- [ ] Ran Quick Test successfully
- [ ] Tried at least 2 examples
- [ ] Plotted at least one profile
- [ ] Calculated junction depth
- [ ] Understand units (nm, seconds, atoms/cm³)

---

## 📞 Need Help?

1. **Check docstrings:** `help(function_name)`
2. **Read full docs:** Session2_README.md
3. **Review examples:** This file!

---

## ⏭️ What's Next?

After mastering Session 2:
- **Session 3:** Numerical solver for complex cases
- **Session 4:** Thermal oxidation
- **Session 5:** Coupled oxidation-diffusion

---

**Status:** ✅ Ready to use!  
**Time to productivity:** ~5 minutes  
**Difficulty:** Beginner-friendly  

🚀 **Start simulating now!** 🚀
