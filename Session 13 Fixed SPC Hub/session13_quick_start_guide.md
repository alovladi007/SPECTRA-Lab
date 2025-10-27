# Session 13: SPC Hub - Quick Start Guide

## 🚀 5-Minute Quick Start

### Step 1: Deploy (2 minutes)

```bash
# Clone repository (if not already done)
git clone https://github.com/yourorg/semiconductorlab.git
cd semiconductorlab

# Deploy Session 13
chmod +x scripts/deploy_session13.sh
./scripts/deploy_session13.sh local
```

**What happens:**
- ✅ Checks system requirements
- ✅ Creates database tables
- ✅ Installs Python/Node dependencies
- ✅ Deploys backend and frontend
- ✅ Runs integration tests
- ✅ Starts Docker containers

---

### Step 2: Access Dashboard (30 seconds)

Open your browser:
```
http://localhost:3000/spc
```

**You'll see:**
- Control chart with sample data
- Process capability metrics
- Alert panel
- Statistics summary

---

### Step 3: Try Demo Scenarios (1 minute)

Use the scenario selector at the top:

1. **In Control** - Normal process (no alerts)
2. **Process Shift** - Detects mean shift after sample 25
3. **Trending** - Detects gradual drift

Click between scenarios to see different SPC patterns!

---

### Step 4: Analyze Your Own Data (1.5 minutes)

#### Python API:

```python
from app.methods.spc import SPCManager, ChartType, DataPoint
from datetime import datetime

# Your data
data = [
    DataPoint(
        timestamp=datetime(2025, 10, 26, 10, i),
        value=your_measurements[i],
        subgroup=f"wafer_{i//5}",
        run_id=f"RUN{i}"
    )
    for i in range(len(your_measurements))
]

# Analyze
manager = SPCManager()
results = manager.analyze_metric(
    metric_name="my_metric",
    data=data,
    chart_type=ChartType.XBAR_R,
    lsl=lower_spec,
    usl=upper_spec
)

# Results
print(f"Cpk: {results['capability']['cpk']:.3f}")
print(f"Alerts: {len(results['alerts'])}")
```

#### REST API:

```bash
curl -X POST http://localhost:8000/api/spc/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "metric_name": "thickness",
    "data_points": [
      {"timestamp": "2025-10-26T10:00:00Z", "value": 150.2, "subgroup": "w1", "run_id": "R1"},
      {"timestamp": "2025-10-26T10:01:00Z", "value": 150.5, "subgroup": "w1", "run_id": "R2"}
    ],
    "chart_type": "xbar_r",
    "lsl": 145.0,
    "usl": 155.0
  }'
```

---

## 📖 Common Use Cases

### Use Case 1: Monitor Sheet Resistance

**Scenario:** Track 4-point probe measurements across wafers

```python
# Example with real measurement data
from app.methods.spc import SPCManager, ChartType, DataPoint
from datetime import datetime, timedelta

# Simulate daily measurements
measurements = [
    # Day 1: Normal
    (98.5, "2025-10-20"), (99.2, "2025-10-20"), (100.1, "2025-10-20"),
    (99.8, "2025-10-20"), (100.3, "2025-10-20"),
    # Day 2: Shift detected
    (103.5, "2025-10-21"), (104.2, "2025-10-21"), (103.8, "2025-10-21")
]

data = []
for i, (value, date_str) in enumerate(measurements):
    data.append(DataPoint(
        timestamp=datetime.fromisoformat(f"{date_str}T{10+i}:00:00"),
        value=value,
        subgroup=f"wafer_{i//5}",
        run_id=f"4PP_RUN_{i+1}",
        metadata={"operator": "Alice", "tool": "4PP_001"}
    ))

# Analyze
manager = SPCManager()
results = manager.analyze_metric(
    metric_name="sheet_resistance",
    data=data,
    chart_type=ChartType.XBAR_R,
    lsl=94.0,  # Ω/sq
    usl=106.0
)

# Check for issues
if len(results['alerts']) > 0:
    print("⚠️  Process out of control!")
    for alert in results['alerts']:
        print(f"  • {alert['message']}")
        print(f"    Actions: {', '.join(alert['suggested_actions'][:2])}")
else:
    print("✅ Process in control")
    print(f"   Cpk: {results['capability']['cpk']:.3f}")
```

---

### Use Case 2: Detect Tool Drift

**Scenario:** Use EWMA to catch early tool degradation

```python
# EWMA is more sensitive to small shifts
results = manager.analyze_metric(
    metric_name="thickness",
    data=your_data,
    chart_type=ChartType.EWMA,  # ← Use EWMA for sensitivity
    lsl=147.0,
    usl=153.0
)

# EWMA detects small shifts (~1σ) that X-bar might miss
```

**When to use EWMA:**
- ✅ Tool wear monitoring
- ✅ Gradual process drift
- ✅ Early warning system
- ❌ NOT for highly variable processes

---

### Use Case 3: Process Capability Study

**Scenario:** Validate process meets 6-sigma requirements

```python
# Collect baseline data (at least 25 subgroups)
baseline_data = collect_data_over_days(days=5)

# Run capability analysis
results = manager.analyze_metric(
    metric_name="critical_dimension",
    data=baseline_data,
    chart_type=ChartType.XBAR_R,
    lsl=spec_lower,
    usl=spec_upper
)

# Interpret
cap = results['capability']
print(f"Process Capability Assessment:")
print(f"  Cp:  {cap['cp']:.3f}  (potential)")
print(f"  Cpk: {cap['cpk']:.3f} (actual)")
print(f"  {cap['interpretation']}")

# Decision matrix
if cap['cpk'] >= 1.67:
    print("✅ Process approved for production")
elif cap['cpk'] >= 1.33:
    print("⚠️  Process acceptable, monitor closely")
else:
    print("❌ Process improvement required")
```

---

## 🎯 Best Practices

### 1. Control Limit Calculation

**Do:**
```python
# Use 20-25 subgroups (100-125 samples minimum)
baseline = data[:125]
limits, _ = xbar_chart.compute_control_limits(baseline)
```

**Don't:**
```python
# Avoid using too few samples
baseline = data[:10]  # ❌ Unreliable limits!
```

---

### 2. Subgrouping Strategy

**Rational Subgroups:**
- ✅ Same wafer, same operator, same shift
- ✅ Minimize within-group variation
- ✅ Maximize between-group variation

**Example:**
```python
# Good subgrouping
DataPoint(value=100.1, subgroup="wafer_1_operator_A_shift_1", ...)

# Poor subgrouping
DataPoint(value=100.1, subgroup="all_wafers", ...)  # ❌ Mixing too much
```

---

### 3. Chart Type Selection

| Use Case | Chart Type | Reason |
|----------|-----------|--------|
| Large shifts (>2σ) | X-bar/R | Simple, interpretable |
| Small shifts (<1σ) | EWMA | More sensitive |
| Sustained drift | CUSUM | Accumulates deviations |
| Binary defects | P-chart | Attributes data |

---

### 4. Alert Response Workflow

```
1. Alert Triggered
   ↓
2. Verify measurement (not instrument error)
   ↓
3. Identify root cause:
   - Recent process changes?
   - Maintenance performed?
   - Material batch change?
   - Operator change?
   ↓
4. Implement corrective action
   ↓
5. Document in ELN
   ↓
6. Recalculate control limits if permanent change
```

---

## 🛠️ Troubleshooting

### Issue: Too Many False Alarms

**Symptoms:**
- Multiple alerts per day
- Process appears stable visually
- Cpk > 1.5 but alerts trigger

**Solutions:**
```python
# 1. Recalculate control limits with more recent data
recent_data = data[-200:]  # Last 200 samples
limits, _ = chart.compute_control_limits(recent_data)

# 2. Increase subgroup size (reduces variability)
chart = XbarRChart(subgroup_size=7)  # Instead of 5

# 3. Use EWMA with higher lambda (less sensitive)
chart = EWMAChart(lambda_=0.3)  # Instead of 0.2
```

---

### Issue: Not Detecting Known Shifts

**Symptoms:**
- Visual shift obvious
- No alerts generated

**Solutions:**
```python
# 1. Check control limits are appropriate
print(f"UCL: {limits.ucl:.2f}, LCL: {limits.lcl:.2f}")
print(f"Range: {limits.ucl - limits.lcl:.2f}")

# 2. Use more sensitive chart
results = manager.analyze_metric(
    ...,
    chart_type=ChartType.EWMA  # More sensitive
)

# 3. Verify data quality
print(f"Samples: {len(data)}")
print(f"Subgroups: {len(set(p.subgroup for p in data))}")
```

---

### Issue: "Insufficient Data" Error

**Symptoms:**
```
{"error": "Insufficient data", "message": "At least 20 data points required"}
```

**Solution:**
```python
# Need minimum 20 points (preferably 100+)
if len(data) < 20:
    print("⚠️  Collecting more data...")
    # Continue data collection
else:
    # Proceed with analysis
    results = manager.analyze_metric(...)
```

---

## 📊 Example Dashboards

### Production Monitoring

```typescript
// Real-time monitoring with auto-refresh
import { SPCDashboard } from '@/app/(dashboard)/spc/page';
import { useState, useEffect } from 'react';

export default function ProductionMonitor() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/api/spc/latest/sheet_resistance');
      setData(await response.json());
    };
    
    fetchData();
    const interval = setInterval(fetchData, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, []);
  
  return data ? <SPCDashboard results={data} onRefresh={() => {}} /> : <Loading />;
}
```

---

### Multi-Metric Overview

```python
# Analyze multiple metrics
metrics = ['sheet_resistance', 'thickness', 'mobility', 'roughness']
results = {}

for metric in metrics:
    data = fetch_metric_data(metric)
    results[metric] = manager.analyze_metric(
        metric_name=metric,
        data=data,
        chart_type=ChartType.XBAR_R
    )

# Summary
for metric, result in results.items():
    status = "OK" if len(result['alerts']) == 0 else "ALERT"
    cpk = result['capability']['cpk']
    print(f"{metric:20s} [{status:5s}] Cpk={cpk:.3f}")
```

Output:
```
sheet_resistance     [OK   ] Cpk=1.850
thickness            [ALERT] Cpk=1.120
mobility             [OK   ] Cpk=1.650
roughness            [OK   ] Cpk=1.430
```

---

## 🎓 Learning Resources

### Tutorials

1. **SPC Basics** (15 minutes)
   - [Video: Introduction to Control Charts](http://localhost:3000/tutorials/spc-basics)
   - Learn: Centerline, UCL, LCL, rule violations

2. **Process Capability** (10 minutes)
   - [Video: Understanding Cp vs Cpk](http://localhost:3000/tutorials/capability)
   - Learn: When to use which index

3. **Advanced Techniques** (20 minutes)
   - [Video: EWMA and CUSUM](http://localhost:3000/tutorials/advanced-spc)
   - Learn: When to use specialized charts

### Interactive Examples

Try these in the dashboard:
1. Navigate to http://localhost:3000/spc
2. Select "Process Shift" scenario
3. Observe alert at sample 26
4. Click "Show Suggested Actions"
5. Review root cause suggestions

---

## 💡 Pro Tips

### Tip 1: Baseline Data Collection

```python
# Collect during stable period (no process changes)
baseline_period = "2025-10-01 to 2025-10-15"
baseline_data = fetch_data(start="2025-10-01", end="2025-10-15")

# Verify stability before using as baseline
preliminary_results = manager.analyze_metric(
    metric_name="baseline_check",
    data=baseline_data,
    chart_type=ChartType.XBAR_R
)

if len(preliminary_results['alerts']) == 0:
    # Good baseline!
    limits = preliminary_results['xbar_limits']
else:
    print("⚠️  Baseline period not stable, collect different period")
```

---

### Tip 2: Automated Reporting

```python
# Daily SPC report
def generate_daily_report():
    yesterday = date.today() - timedelta(days=1)
    data = fetch_data(date=yesterday)
    
    results = manager.analyze_metric(
        metric_name="daily_metric",
        data=data,
        chart_type=ChartType.XBAR_R
    )
    
    # Email if alerts
    if len(results['alerts']) > 0:
        send_email(
            to="process_engineers@company.com",
            subject=f"SPC Alert - {yesterday}",
            body=format_alert_email(results['alerts'])
        )
```

---

### Tip 3: Control Limit Versioning

```python
# Save control limits to database for audit trail
from app.models import SPCControlLimit

limit_record = SPCControlLimit(
    metric="sheet_resistance",
    chart_type="xbar_r",
    ucl=limits.ucl,
    lcl=limits.lcl,
    centerline=limits.cl,
    sigma=limits.sigma,
    valid_from=datetime.now(),
    computed_from_runs=[p.run_id for p in baseline_data]
)

db.add(limit_record)
db.commit()
```

---

## 🔗 Next Steps

1. **Integrate with your data pipeline**
   - Connect to your measurement systems
   - Automate data collection
   - Set up scheduled analysis

2. **Configure alerts**
   - Email notifications
   - Slack integration
   - SMS for critical alerts

3. **Train your team**
   - SPC interpretation workshop
   - Alert response procedures
   - Root cause analysis techniques

4. **Advance to Session 14**
   - Machine learning for predictive analytics
   - Virtual metrology models
   - Anomaly prediction

---

## 📞 Support

**Questions?**
- 📧 Email: support@semiconductorlab.com
- 💬 Slack: #spc-support
- 📖 Docs: http://localhost:3000/docs/spc

**Found a bug?**
- 🐛 GitHub: https://github.com/yourorg/semiconductorlab/issues
- Label: `session-13` + `spc`

---

**Happy SPC Monitoring! 📊✨**

*Remember: Statistical Process Control is a tool for continuous improvement, not just defect detection. Use it wisely!*
