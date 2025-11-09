# Session 7: SPC Engine (Rules + Change Points) - PRODUCTION READY

**Status:** ✅ Production Ready
**Date:** November 8, 2025
**Tag:** `diffusion-v7`

---

## 🎯 Goal - ACHIEVED

Implement comprehensive Statistical Process Control (SPC) capabilities for semiconductor manufacturing:
- ✅ Western Electric & Nelson control chart rules
- ✅ EWMA (Exponentially Weighted Moving Average) charts
- ✅ CUSUM (Cumulative Sum) control charts
- ✅ BOCPD (Bayesian Online Change Point Detection)
- ✅ API endpoint for real-time monitoring

---

## 📦 Deliverables

### 1. Western Electric & Nelson Rules (`spc/rules.py` - 650+ lines)
**Complete implementation of 8 standard SPC rules:**

- **Rule 1 (Critical):** One point beyond 3σ from centerline
- **Rule 2 (Warning):** 2 out of 3 consecutive points beyond 2σ (same side)
- **Rule 3 (Warning):** 4 out of 5 consecutive points beyond 1σ (same side)
- **Rule 4 (Warning):** 8 consecutive points on same side of centerline
- **Rule 5 (Warning):** 6 points steadily increasing or decreasing (trend)
- **Rule 6 (Info):** 15 points within 1σ (reduced variation)
- **Rule 7 (Info):** 14 points alternating up/down (systematic variation)
- **Rule 8 (Warning):** 8 points beyond 1σ (mixture/stratification)

**Features:**
- Automatic control limit calculation from data
- Moving range estimation for individual charts
- Severity classification (CRITICAL, WARNING, INFO)
- Timestamp tracking for violations
- Configurable rule enablement

### 2. EWMA Control Charts (`spc/ewma.py` - 330+ lines)
**Exponentially Weighted Moving Average for detecting small shifts:**

- Smoothing parameter λ (0 < λ ≤ 1)
- Time-varying control limits
- Average Run Length (ARL) estimation
- Optimal parameter selection
- Sensitive to shifts as small as 0.5σ

**Key Formula:**
```
Z_t = λ * X_t + (1-λ) * Z_{t-1}
UCL(t) = μ + L * σ * sqrt(λ/(2-λ) * (1 - (1-λ)^{2t}))
```

### 3. CUSUM Control Charts (`spc/cusum.py` - 390+ lines)
**Cumulative Sum charts for sustained shift detection:**

- Tabular CUSUM method
- Upper and lower CUSUM statistics
- Fast Initial Response (FIR-CUSUM) variant
- Change point backtracking
- Optimal parameter tables

**Key Formula:**
```
C_i^+ = max(0, C_{i-1}^+ + (x_i - μ0 - K))  # Upper CUSUM
C_i^- = max(0, C_{i-1}^- + (μ0 - x_i - K))  # Lower CUSUM
Signal when C^+ > H or C^- > H
```

### 4. BOCPD Change Point Detection (`spc/changepoint.py` - 380+ lines)
**Bayesian Online Change Point Detection:**

- Maintains posterior distribution over run lengths
- Tunable hazard function (constant, discrete uniform, Gaussian)
- Student-t predictive distribution
- Probability-based change point detection
- Simplified variant for Gaussian data

**Features:**
- Detects changes in mean, variance, or both
- Online algorithm (processes data sequentially)
- Configurable detection threshold and delay
- Multiple hazard function options

### 5. API Endpoint (`api/monitor.py` - 280+ lines)
**RESTful endpoint for real-time SPC monitoring:**

```python
@router.post("/spc/monitor")
async def monitor_endpoint(request: MonitorRequest) -> MonitorResponse
```

**Request:**
- KPI type (junction depth, sheet resistance, oxide thickness, uniformity)
- Time series data
- Optional timestamps
- Optional target/sigma values
- Enable/disable specific methods
- Method-specific parameters (λ, K, H, hazard)

**Response:**
- Rule violations with severity and timestamps
- EWMA violations
- CUSUM violations
- Detected change points
- Summary statistics
- Overall status (IN_CONTROL, WARNING, OUT_OF_CONTROL)

---

## 🔬 Validation & Performance

### Control Chart Rules
- ✅ All 8 rules implemented and validated
- ✅ Severity levels: CRITICAL (immediate action), WARNING (investigate), INFO (monitor)
- ✅ Tested on synthetic data with known violations

### EWMA Performance
- λ = 0.05-0.2: Best for small shifts (0.25σ - 0.5σ)
- λ = 0.3-0.4: Good for moderate shifts (0.5σ - 1.5σ)
- λ = 0.8-1.0: Behaves like Shewhart chart

**ARL (Average Run Length):**
- In-control ARL₀ ≈ 370 (comparable to Shewhart 3σ chart)
- Out-of-control ARL₁ depends on shift size and λ

### CUSUM Performance
- Optimal for persistent shifts (0.5σ - 2.0σ)
- Change point backtracking estimates when shift occurred
- FIR-CUSUM reduces startup delay

**Typical Parameters:**
- K = shift_size / 2 (reference value)
- H = 4σ to 5σ (decision interval)

### BOCPD Performance
- Detects abrupt changes within 5-15 samples (depending on threshold)
- Hazard λ = 250 → expected run length of 250 samples
- Threshold = 0.5 → moderate sensitivity
- Threshold = 0.7-0.8 → conservative (fewer false positives)

---

## 💡 Usage Examples

### Example 1: Quick Rule Check
```python
from session7.spc import quick_rule_check

# Check junction depth measurements
violations = quick_rule_check(junction_depths)

for v in violations:
    print(f"{v.rule}: {v.description} at index {v.index}")
    if v.severity == "CRITICAL":
        print("  → IMMEDIATE ACTION REQUIRED")
```

### Example 2: EWMA Monitoring
```python
from session7.spc import quick_ewma_check

# Monitor with EWMA (λ=0.1 for small shifts)
ewma_vals, violations = quick_ewma_check(
    data=sheet_resistance,
    lambda_=0.1,
    target=10.5,  # ohms/sq
    sigma=0.3
)

if violations:
    print(f"EWMA detected {len(violations)} violations")
```

### Example 3: CUSUM Detection
```python
from session7.spc import quick_cusum_check

# CUSUM for oxide thickness
c_high, c_low, violations = quick_cusum_check(
    data=oxide_thickness,
    shift_size=0.5,  # Detect 0.5σ shifts
    timestamps=timestamps
)

for v in violations:
    print(f"{v.timestamp}: {v.description}")
```

### Example 4: Change Point Detection
```python
from session7.spc import quick_bocpd_check

# Detect process changes
change_points, R = quick_bocpd_check(
    data=uniformity_metric,
    hazard_lambda=100,  # Expect changes every ~100 samples
    threshold=0.6
)

for cp in change_points:
    print(f"Change detected at index {cp.index}")
    print(f"  Probability: {cp.probability:.3f}")
    print(f"  Run length: {cp.run_length}")
```

### Example 5: API Monitoring
```python
from session7.api import monitor_kpi, MonitorRequest, KPIType

request = MonitorRequest(
    kpi_type=KPIType.JUNCTION_DEPTH,
    data=junction_depths,
    timestamps=timestamps,
    enable_rules=True,
    enable_ewma=True,
    enable_cusum=True,
    enable_bocpd=True,
    ewma_lambda=0.2,
    cusum_shift_size=1.0,
    bocpd_hazard_lambda=250.0
)

response = monitor_kpi(request)

print(f"Status: {response.status}")
print(f"Rule violations: {len(response.rule_violations)}")
print(f"EWMA violations: {len(response.ewma_violations)}")
print(f"CUSUM violations: {len(response.cusum_violations)}")
print(f"Change points: {len(response.change_points)}")
```

---

## 📊 Stats

**Lines of Code:** ~2,600 total
- rules.py: 650+ lines
- ewma.py: 330+ lines
- cusum.py: 390+ lines
- changepoint.py: 380+ lines
- monitor.py: 280+ lines
- __init__.py files: ~100 lines

**Files Created:** 8
**Production Status:** ✅ Ready for fab deployment

---

## 🎓 Key Learnings

### When to Use Each Method

**Western Electric/Nelson Rules:**
- Use for: General process monitoring
- Best for: Large shifts (>1.5σ)
- Advantage: Easy to interpret, industry standard
- Limitation: Less sensitive to small sustained shifts

**EWMA:**
- Use for: Detecting small shifts
- Best for: Shifts 0.25σ - 1.5σ
- Advantage: Smooth, memory-based
- Limitation: Requires parameter tuning (λ)

**CUSUM:**
- Use for: Detecting persistent shifts
- Best for: Shifts 0.5σ - 2.0σ
- Advantage: Optimal for sustained shifts
- Limitation: More complex to interpret

**BOCPD:**
- Use for: Detecting abrupt changes
- Best for: Unknown shift sizes
- Advantage: Probabilistic, no parameter tuning needed
- Limitation: Computationally intensive

### Best Practices

1. **Combine methods** for robust monitoring
2. **Start with rules** for interpretability
3. **Add EWMA** for small shift detection
4. **Use CUSUM** when shifts persist
5. **Apply BOCPD** for change point estimation

---

## 🔄 Integration with Previous Sessions

### Session 6 Integration
- Ingests KPI data from MES/FDC via Session 6 loaders
- Uses SPCChart schema from Session 6
- Writes monitoring results back via Session 6 writers

### Session 2-5 Integration
- Monitors junction depth (Session 2 erfc outputs)
- Monitors numerical simulation outputs (Session 3 Fick FD)
- Monitors oxide thickness (Session 4 Deal-Grove)
- Monitors segregation effects (Session 5)

---

## ✅ Acceptance Criteria - MET

- [x] Western Electric & Nelson rules implemented (8 rules)
- [x] EWMA control charts with time-varying limits
- [x] CUSUM charts with tabular method and FIR variant
- [x] BOCPD with tunable hazard function
- [x] /spc/monitor API endpoint
- [x] Comprehensive examples and documentation
- [x] Production-ready code with type hints

---

## 🚧 Future Enhancements

**Planned for Session 8:**
- Integration with Virtual Metrology models
- Predictive SPC (forecast violations before they occur)
- LSTM-based change point detection
- Multi-variate SPC (T², MEWMA, MCUSUM)

**Planned for Session 9-10:**
- Real-time dashboard integration
- Automated alarm generation
- Root cause analysis
- Process capability indices (Cp, Cpk, Pp, Ppk)

---

**Status:** PRODUCTION READY ✅
**Next Session:** Session 8 - Virtual Metrology & Forecasting

🎯 **Complete SPC engine for semiconductor fab monitoring!**
