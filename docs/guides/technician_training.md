# Lab Technician Training Guide

## SemiconductorLab Platform - Electrical Characterization

**Version:** 1.0  
**Date:** November 2025  
**Duration:** 2-day course  
**Prerequisites:** Basic semiconductor knowledge, lab safety certification

-----

## 📚 Table of Contents

1. [Platform Overview](#platform-overview)
1. [Safety First](#safety-first)
1. [Four-Point Probe Measurements](#four-point-probe-measurements)
1. [Hall Effect Measurements](#hall-effect-measurements)
1. [Data Analysis & Interpretation](#data-analysis--interpretation)
1. [Troubleshooting](#troubleshooting)
1. [Best Practices](#best-practices)
1. [Certification Quiz](#certification-quiz)

-----

## Platform Overview

### What is SemiconductorLab?

SemiconductorLab is an integrated characterization platform that helps you:

- **Measure** electrical properties of semiconductor samples
- **Analyze** data automatically with advanced algorithms
- **Track** sample history and ensure compliance
- **Report** results professionally for customers and audits

### Key Features You’ll Use

✅ **Automatic Data Collection** - No manual transcription  
✅ **Real-time Validation** - Catch errors immediately  
✅ **Quality Checks** - Automatic outlier detection  
✅ **Report Generation** - Professional PDFs in seconds  
✅ **Compliance** - Full traceability for ISO 17025

-----

## Safety First

### ⚠️ CRITICAL SAFETY RULES

**BEFORE EVERY MEASUREMENT:**

1. **Inspect the sample**
- No cracks or damage
- No conductive debris on contacts
- Proper mounting on chuck
1. **Check instrument settings**
- Compliance limits set correctly
- Current/voltage within sample ratings
- Emergency stop accessible
1. **Verify connections**
- All cables secure
- No exposed conductors
- Ground connection verified

### Electrical Hazards

|Hazard             |Risk Level  |Mitigation                            |
|-------------------|------------|--------------------------------------|
|High voltage (>50V)|⚠️ **HIGH**  |Use insulated probes, no jewelry      |
|High current (>1A) |⚠️ **MEDIUM**|Check wire ratings, watch for heating |
|Magnetic field     |⚠️ **LOW**   |Remove magnetic cards, pacemaker alert|

### Emergency Procedures

**If you smell burning or see smoke:**

1. Press **EMERGENCY STOP** button (red)
1. Disconnect power at breaker
1. Call supervisor immediately
1. Do not touch sample until cool

**If you receive a shock:**

1. Seek medical attention immediately
1. Report to supervisor
1. Tag equipment “DO NOT USE”

-----

## Four-Point Probe Measurements

### What Does 4PP Measure?

Four-point probe measures **sheet resistance** (Rs), which tells you:

- How conductive your sample is
- If doping is uniform across a wafer
- Whether your deposition/implantation was successful

**Key Equation:**

Sheet Resistance (Rs) = Resistivity (ρ) / Thickness (t)
Units: Ω/sq (ohms per square)

### Sample Preparation

#### ✅ DO:

- Clean sample with isopropanol
- Check contact pads are clear
- Verify sample ID matches paperwork
- Note any visible defects in log

#### ❌ DON’T:

- Touch contact pads with bare hands
- Use damaged probes
- Force probes onto sample
- Measure cracked samples

### Step-by-Step Procedure

#### 1. Login to System

1. Open browser: http://semiconductorlab.local
2. Enter your credentials
3. Select "Electrical" → "Four-Point Probe"

#### 2. Load Sample Information

1. Scan sample barcode or enter ID
2. Verify displayed info matches paperwork:
   - Wafer ID
   - Lot number
   - Material type
   - Thickness
3. Click "Continue"

#### 3. Configure Measurement

**Standard Settings (most samples):**

- Test Current: **1.0 mA**
- Configurations: **4** (recommended)
- Temperature: **300 K** (room temp)
- Outlier Rejection: **Enabled**

**When to Change Settings:**

|Sample Type               |Current|Notes         |
|--------------------------|-------|--------------|
|High resistance (>1 kΩ/sq)|0.1 mA |Reduce noise  |
|Low resistance (<10 Ω/sq) |10 mA  |Improve signal|
|Thin films (<100 nm)      |0.5 mA |Avoid damage  |

#### 4. Probe Placement

**Van der Pauw Method (arbitrary shapes):**

     A ●─────────● B
       │         │
       │  SAMPLE │
       │         │
     D ●─────────● C

Contacts at corners, avoid edges
Minimum 5mm from sample edge

**Tips:**

- ✅ Contacts should be small (<1 mm)
- ✅ Apply gentle, even pressure
- ✅ Check for good contact (low resistance)
- ❌ Don’t scratch sample surface

#### 5. Run Measurement

1. Click "Start Measurement"
2. Watch live readings for stability
3. Wait for all configurations to complete
4. System will automatically:
   - Calculate sheet resistance
   - Check for outliers
   - Validate contact quality

**What You’ll See:**

|Display          |What It Means   |Good/Bad     |
|-----------------|----------------|-------------|
|Voltage: 125 mV  |Measured voltage|✅ Stable     |
|Current: 1.00 mA |Applied current |✅ Correct    |
|Resistance: 125 Ω|V/I calculation |✅ Reasonable |
|CV: 1.8%         |Variability     |✅ <5% is good|

#### 6. Review Results

**What to Check:**

1. **Sheet Resistance Value**
- Is it in expected range?
- Compare to spec sheet
1. **Statistics**
- CV% <5% → Good
- CV% 5-10% → Acceptable
- CV% >10% → Check contacts
1. **Quality Checks**
- ✅ Contact check PASSED
- ✅ No outliers (or <2)
- ✅ Temperature stable
1. **Wafer Map** (if mapped)
- Uniformity <5% is excellent
- Look for edge effects
- Check for patterns (equipment issues)

#### 7. Save & Report

1. Review data
2. Add notes (if needed):
   - "Sample had slight discoloration"
   - "Repeated measurement, first had bad contact"
3. Click "Save Results"
4. Generate PDF report
5. Print and attach to sample traveler

### Common 4PP Issues & Solutions

|Problem             |Likely Cause      |Solution                     |
|--------------------|------------------|-----------------------------|
|Very high CV (>15%) |Poor contacts     |Clean probes, reposition     |
|Negative resistance |Probe reversed    |Check polarity               |
|Contact check failed|Contamination     |Clean sample, retry          |
|Asymmetric readings |Damaged sample    |Note in log, consult engineer|
|No reading          |Disconnected cable|Check all connections        |

-----

## Hall Effect Measurements

### What Does Hall Measure?

Hall effect tells you **WHO** the carriers are (not just HOW MANY):

- **Carrier type:** n-type (electrons) or p-type (holes)
- **Carrier concentration:** How many carriers (n or p)
- **Mobility:** How fast carriers move (affects conductivity)

**Why This Matters:**

- Verify doping type (n vs p)
- Calculate conductivity
- Quality check for semiconductor processing

### The Hall Effect (Simple Explanation)

When you apply a magnetic field perpendicular to current flow, carriers deflect to one side:

      MAGNETIC FIELD (B) ↑
            ┌─────────┐
   I →      │    +    │      ← Holes deflect this way
            │ SAMPLE  │
   I →      │    -    │      ← Electrons deflect opposite
            └─────────┘
                ↕
         Hall Voltage (VH)

**Key Point:** The **SIGN** of the Hall voltage tells you n-type (negative) or p-type (positive)!

### Sample Requirements

✅ **Good samples:**

- Uniform thickness
- Known from 4PP measurement
- Clean contacts
- <1 kΩ/sq sheet resistance

⚠️ **Difficult samples:**

- Very high resistance (>10 kΩ/sq)
- Very thin (<10 nm)
- Compensated (mixed n/p)

### Step-by-Step Procedure

#### 1. Login & Load Sample

Same as 4PP - scan barcode, verify info.

**IMPORTANT:** If you did 4PP first, the system will ask:

"Use sheet resistance from 4PP measurement?"
→ Click YES (this is best practice)

#### 2. Configure Measurement

**Standard Settings:**

- Measurement Type: **Multi-field** (preferred)
- Test Current: **1.0 mA**
- Field Range: **-1.0 to +1.0 Tesla**
- Number of Points: **11**
- Temperature: **300 K**

**Multi-field vs Single-field:**

|Method      |Accuracy     |Time   |When to Use                     |
|------------|-------------|-------|--------------------------------|
|Multi-field |⭐⭐⭐ Excellent|3-5 min|**DEFAULT** - eliminates offsets|
|Single-field|⭐⭐ Good      |1-2 min|Quick screening only            |

#### 3. Position Sample

**Hall Bar Geometry (preferred):**

   I+  ●─────────────────────● I-
       │                     │
   VH+ ●     CURRENT FLOW    │
       │    (horizontal)     │
   VH- ●                     │
       │_____________________│
       
Magnetic field points UP (out of page)

**Important:**

- Current terminals on long sides
- Hall voltage measured perpendicular to current
- Sample centered in magnet gap

#### 4. Run Measurement

1. Verify magnet is ON (LED indicator)
2. Check field calibration (should be recent)
3. Click "Start Measurement"
4. Watch live Hall voltage vs field
5. Graph should be LINEAR

**What You’ll See:**

|Reading         |Example   |Meaning               |
|----------------|----------|----------------------|
|Hall Voltage    |-2.5 μV   |Small voltage (normal)|
|Magnetic Field  |+0.5 T    |Current field strength|
|Hall Coefficient|-125 cm³/C|Negative = n-type     |

**Good Signs:**

- ✅ Linear plot (R² > 0.99)
- ✅ Clear sign (all + or all -)
- ✅ Low noise (CV < 5%)

**Bad Signs:**

- ❌ Curved plot (saturation effects)
- ❌ Sign changes (contamination)
- ❌ Very noisy (check connections)

#### 5. Interpret Results

**Main Results:**

1. **Carrier Type**
   
   n-type → Electrons (most semiconductors)
   p-type → Holes (boron-doped silicon)
1. **Carrier Concentration**
   
   Example: 5.0×10¹⁸ cm⁻³
   
   Interpretation:
   < 10¹⁶: Lightly doped
   10¹⁷-10¹⁹: Moderately doped
   > 10²⁰: Heavily doped
1. **Mobility**
   
   Example: 1200 cm²/(V·s) for n-Si
   
   Higher mobility = Better quality
   
   Typical values:
   - n-Silicon: 1000-1500
   - p-Silicon: 300-500
   - GaAs: 5000-8500
   - Graphene: 10,000+

#### 6. Quality Assessment

System automatically scores 0-100:

|Score|Level       |Action                           |
|-----|------------|---------------------------------|
|>90  |Excellent ✅ |Accept, no issues                |
|70-90|Good ✅      |Accept, note any warnings        |
|50-70|Acceptable ⚠️|Review warnings, consult engineer|
|<50  |Poor ❌      |Repeat measurement or reject     |

**Common Warnings:**

- “High variability (CV >5%)” → Check contacts, reduce noise
- “Mobility exceeds expected” → Verify sheet resistance input
- “Low R²” → Check for magnetic saturation, reduce field range

#### 7. Save & Report

Same as 4PP - review, annotate, save, generate PDF.

-----

## Data Analysis & Interpretation

### Understanding Your Results

#### Four-Point Probe Output

**Example Report:**

SHEET RESISTANCE: 125.3 Ω/sq
  ± 2.1 Ω/sq (1.68%)
  
RESISTIVITY: 0.627 Ω·cm
  (Sample thickness: 500 μm)
  
STATISTICS:
  Mean: 125.3 Ω/sq
  Std Dev: 2.1 Ω/sq
  CV%: 1.68%
  Measurements: 4
  Outliers: 0
  
QUALITY:
  ✅ Contact resistance: PASSED
  ✅ Variability: EXCELLENT (<2%)
  ✅ Temperature: STABLE (300 K)

**What Does This Mean?**

✅ **Good Result:**

- Sheet resistance in expected range
- Low variability (CV <2%)
- No quality issues

**Next Steps:**

- Compare to specification
- Check against previous wafers
- Release to next process step

#### Hall Effect Output

**Example Report:**

CARRIER TYPE: n-type
  
HALL COEFFICIENT: -125.3 cm³/C
  R² = 0.9987 (excellent fit)
  
CARRIER CONCENTRATION: 4.98×10¹⁸ cm⁻³
  
HALL MOBILITY: 1185 cm²/(V·s)
  
CONDUCTIVITY: 94.2 S/cm
  
SHEET RESISTANCE: 0.201 Ω/sq
  (from 4PP measurement)
  
QUALITY: EXCELLENT (95.5/100)
  ✅ Linear regression: R² > 0.99
  ✅ Carrier concentration reasonable
  ✅ Mobility within expected range

**What Does This Mean?**

✅ **Good Result:**

- Clear n-type identification
- Carrier concentration matches target (5×10¹⁸)
- Mobility typical for n-Silicon
- Excellent measurement quality

### When to Call an Engineer

🚨 **STOP and consult engineer if:**

1. **Unexpected carrier type**
- Spec says n-type, you measured p-type
- ACTION: Do not proceed, notify engineer
1. **Extreme values**
- Sheet resistance >10× expected
- Carrier concentration >10²² cm⁻³
- Mobility >5000 cm²/(V·s) for Silicon
- ACTION: Repeat measurement, if same → engineer
1. **Quality score <50**
- Multiple warnings
- Poor linearity (R² <0.95)
- High variability (CV >15%)
- ACTION: Check setup, repeat once, if poor → engineer
1. **Physically damaged sample**
- Cracks, chips, contamination
- ACTION: Photo, log, engineer review
1. **Equipment errors**
- “Connection lost”
- “Calibration expired”
- “Instrument error”
- ACTION: Do not force, notify engineer/maintenance

-----

## Troubleshooting

### Problem: High Measurement Variability

**Symptoms:** CV% >10%, large std dev

**Checklist:**

- [ ] Clean probes with isopropanol
- [ ] Check probe tip condition (no damage)
- [ ] Verify sample surface is clean
- [ ] Ensure probes contact pads fully
- [ ] Check cable connections
- [ ] Verify room temperature stable

**If still high:** Note in system, consult engineer

### Problem: Contact Check Failed

**Symptoms:** “Contact resistance too high” warning

**Checklist:**

- [ ] Clean contact pads with isopropanol
- [ ] Remove oxide (if sample allows)
- [ ] Check probe tips for damage
- [ ] Verify probes are properly aligned
- [ ] Try gentle probe pressure increase

**If still failing:** Sample may need re-metallization

### Problem: Instrument Not Responding

**Symptoms:** “Connection timeout”, “Instrument not found”

**Checklist:**

- [ ] Check instrument power (green LED)
- [ ] Verify USB/GPIB cable connected
- [ ] Restart instrument (power cycle)
- [ ] Check software connection settings
- [ ] Try different USB port

**If still failing:** Create maintenance ticket

### Problem: Hall Voltage Zero or Near-Zero

**Symptoms:** Hall coefficient ~0, type = “unknown”

**Possible Causes:**

1. **No magnetic field**
- Check magnet power supply
- Verify field calibration
1. **Compensated semiconductor**
- Nearly equal n and p carriers
- This is rare but possible
- Consult engineer
1. **Shorted contacts**
- Check Hall probes not touching
- Verify probe placement

### Problem: Wafer Map Shows Pattern

**Symptoms:** Systematic variation across wafer

**Interpretation:**

┌─────────────────┐
│  Low   →  High  │  → Gradient (deposition issue)
│                 │
└─────────────────┘

┌─────────────────┐
│ High  Center Low│  → Edge effect (expected)
│ High         Low│
└─────────────────┘

┌─────────────────┐
│ X     X     X   │  → Periodic pattern (chuck issue)
│   X     X     X │
└─────────────────┘

**Action:**

- Note pattern type in log
- Take screenshot
- Report to process engineer
- May indicate process equipment issue

-----

## Best Practices

### Daily Startup

✅ **Every Morning:**

1. Check instrument status LEDs (all green)
1. Review overnight calibration logs
1. Run diagnostic test (built into system)
1. Measure reference standard
1. Log all checks in system

### Sample Handling

✅ **DO:**

- Handle by edges only
- Use vacuum wand or tweezers
- Store in clean room environment
- Keep in closed carriers

❌ **DON’T:**

- Touch device area
- Stack samples
- Expose to humidity
- Leave unprotected

### Data Integrity

✅ **Always:**

- Double-check sample ID before measurement
- Annotate unusual observations
- Report equipment issues immediately
- Sign off on measurements
- Never delete raw data

### Efficiency Tips

💡 **Work Smarter:**

1. **Batch similar samples**
- Set up once, measure many
- Saves setup time
1. **Use templates**
- Common recipes pre-configured
- One-click measurement
1. **Review while measuring**
- Check previous results during current run
- Multi-task efficiently
1. **Monitor trends**
- If 5 samples from same lot are consistent
- Spot check remaining samples

### End of Day

✅ **Before Leaving:**

- [ ] Log out of system
- [ ] Clean probes (isopropanol, gentle)
- [ ] Cover instruments
- [ ] Check data uploaded to server
- [ ] File printed reports
- [ ] Note any issues in logbook

-----

## Certification Quiz

### Section 1: Safety (Must get 100%)

1. **What should you do if you smell burning during measurement?**
- a) Continue and note in log
- b) Press emergency stop, disconnect power, call supervisor ✓
- c) Take a photo for the log
- d) Reduce the test current
1. **Before starting a measurement, you must:**
- a) Check instrument is on
- b) Verify compliance limits
- c) Inspect sample for damage
- d) All of the above ✓
1. **If you receive an electrical shock, you should:**
- a) Seek medical attention immediately ✓
- b) Continue working
- c) Report at end of shift
- d) Reset the instrument

### Section 2: Four-Point Probe

1. **What does sheet resistance measure?**
- a) Resistivity divided by thickness ✓
- b) Total resistance of sample
- c) Contact resistance
- d) Carrier mobility
1. **Good CV% for 4PP is:**
- a) <5% ✓
- b) <20%
- c) <50%
- d) Any value is fine
1. **If contact check fails, you should:**
- a) Increase test current
- b) Clean contacts and retry ✓
- c) Skip the check
- d) Call IT support

### Section 3: Hall Effect

1. **Negative Hall coefficient means:**
- a) p-type semiconductor
- b) n-type semiconductor ✓
- c) Measurement error
- d) No carriers
1. **Multi-field measurement is preferred because:**
- a) It’s faster
- b) It eliminates offset voltages ✓
- c) It uses less current
- d) It’s easier to set up
1. **Typical mobility for n-type Silicon is:**
- a) 50 cm²/(V·s)
- b) 500 cm²/(V·s)
- c) 1200 cm²/(V·s) ✓
- d) 10,000 cm²/(V·s)

### Section 4: Troubleshooting

1. **If quality score is <50, you should:**
- a) Accept results anyway
- b) Repeat measurement, then consult engineer if still poor ✓
- c) Delete the data
- d) Increase test current

**Passing Score: 9/10 (90%)**  
**Safety Questions: Must get 100%**

-----

## Quick Reference Card

*Print this page and laminate for lab bench*

### Four-Point Probe

|Setting|Default|Range    |
|-------|-------|---------|
|Current|1.0 mA |0.1-10 mA|
|Configs|4      |2-8      |
|Temp   |300 K  |77-400 K |

**Good Result:** CV <5%, Contact check ✅

### Hall Effect

|Setting|Default    |Range       |
|-------|-----------|------------|
|Type   |Multi-field|Single/Multi|
|Current|1.0 mA     |0.1-10 mA   |
|Field  |±1.0 T     |±0.1-2.0 T  |
|Points |11         |5-20        |

**Good Result:** R² >0.99, Quality >70

### Emergency Contacts

|Issue              |Contact       |
|-------------------|--------------|
|Equipment emergency|Extension 911 |
|Process engineer   |Extension 1234|
|IT/Software        |Extension 5678|
|Maintenance        |Extension 3456|

### Daily Checks

- [ ] Instrument LEDs green
- [ ] Reference standard measured
- [ ] Probes clean
- [ ] Room temp 20-25°C

-----

## Congratulations!

You’ve completed the SemiconductorLab Electrical Characterization training.

**Next Steps:**

1. Take certification quiz with supervisor
1. Perform 5 measurements under supervision
1. Review and sign training checklist
1. Receive lab access badge

**Welcome to the team!** 🎉

-----

*Document Version: 1.0*  
*Last Updated: November 2025*  
*Contact Training Coordinator for questions*