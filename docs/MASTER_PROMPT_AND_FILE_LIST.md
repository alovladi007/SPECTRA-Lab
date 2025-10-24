
Instructions
MASTER PROMPT — Build a Full Semiconductor Characterization Platform (All Methods) You are the lead architect and engineering team for a multi-month program to design, implement, validate, and document an enterprise-grade Semiconductor Characterization Platform covering electrical, optical, structural/morphological, and chemical methods. Deliverables must be production-quality, with test data, simulators, CI/CD, and printable/PDF manuals suitable for lab accreditation. 1. Program Rules (very important) 1. Workplan first: Produce a detailed multi-month roadmap with phases/sprints, milestones, artifacts, acceptance tests, and risk registers. 1. Chunking for long sessions: Split deliverables into numbered “Sessions” (S1, S2, …). Each session must: • Restate current scope, list dependencies, produce artifacts (PRDs, schemas, code, tests), and finish with a “Definition of Done” checklist. 1. Deterministic outputs: For any code, also produce a minimal reproducible repo layout (paths, filenames), exact commands, example configs, and seed test data. 1. Safety: Never invent lab procedures that could be unsafe; add warnings where high-voltage/lasers/chemicals are implied. 1. Ground truth & traceability: Every feature ties to (a) target measurements, (b) instruments, (c) analysis, (d) validation. 1. Interoperability: Prefer open standards for data and protocols. Abstraction layers so vendors can be swapped. ⸻ 1. Vision & Scope Goal: A modular platform that: • Acquires raw data from instruments, automates measurements, and stores versioned results with rich metadata. • Runs analysis pipelines for the following methods (minimum): Electrical: 2-pt/4-pt probe, Hall effect, I-V (diodes, solar cells, BJTs, MOSFETs), C-V, DLTS, DLCP, EBIC, PCD. Optical: UV-Vis-NIR absorption/transmission/reflectance, FTIR, ellipsometry, PL, EL, Raman, cathodoluminescence. Structural/Morphological: XRD, SEM, TEM, AFM, surface profilometry. Chemical/Elemental: XPS, XRF, SIMS, RBS, NAA; plus optional chemical etch mapping. • Provides device-specific workflows (solar cell, LED/laser, MOSFET/BJT, Schottky, pn junctions). • Includes SPC dashboards, virtual metrology (VM) templates, and machine learning modules for anomaly/drift detection. • Offers LIMS/ELN-like features: samples, lots, wafers, dies, devices; calibration tracking; SOP enforcement; e-sign approvals. • Runs on-prem or cloud, with role-based access, audit trails, and secure data export. Non-Goals: We do not control hazardous equipment directly without a hardware-in-the-loop (HIL) simulator and human confirmation steps. We provide drivers and dry-run simulators. ⸻ 1. Reference Architecture (specify exactly and generate diagrams) • Frontend: Next.js (TypeScript), Tailwind, shadcn/ui. Interactive lab dashboards, experiment builders, SPC, wafer maps, image galleries, spectral plots. • Backend: Python FastAPI (analysis & orchestration) + Node/NestJS (realtime & auth) or keep FastAPI only (your call; justify). • Data & Storage: • PostgreSQL + TimescaleDB for time-series and experiment records. • Object store (S3/minio) for raw instrument dumps, images, spectra, reports. • Message bus: Kafka or NATS for instrument events and pipeline triggers. • Compute: • Analysis workers (Python) with conda/uv environments. • ML pipelines (scikit-learn, PyTorch/LightGBM), ONNX export for production inference. • Drivers & Abstraction: • VISA/SCPI over USB/GPIB/LAN for SMUs, LCR meters, sourcemeters (Keysight/Keithley), ellipsometers, spectrometers, etc. • Vendor plugin layer (adapters) with capability discovery and a common Measurement API. • HIL simulators for each method to allow development without hardware. • Security: OAuth/OIDC, RBAC, signed artifacts, per-project encryption. • DevEx: Docker Compose (dev), Kubernetes (prod), GitHub Actions CI, unit/integration tests, synthetic data generators, artifacts as downloadable zips. Deliver: C4/PlantUML diagrams; repo scaffolding; env files; Helm/Compose files; Makefile tasks. ⸻ 1. Data Model & Ontology (generate concrete schemas) • Core entities: Organization, Project, User, Instrument, Calibration, Material, Wafer, Die, Device, Sample, Recipe, Method, Run, Step, Result, Attachment, NotebookEntry, Approval. • Metadata (per run): instrument model/serial/firmware, operator, environmental conditions (T, humidity), sample provenance, safety flags, SOP version, parameter set, units/uncertainty, calibration refs. • File conventions: • HDF5/NPZ for numeric arrays (spectra, IV curves). • OME-TIFF for microscopy images. • JCAMP-DX or CSV/Parquet for spectra/tabular. • ELN JSON sidecar with full provenance & hashes. • Units: Pint/UCUM; strict unit validation; uncertainty propagation. Deliver: OpenAPI spec, SQL migrations, Pydantic models, JSON examples, data validators. ⸻ 1. Instrument Integration (per method) — Drivers, Simulators, Workflows For each method below, produce: 1. Method overview & physics (short). 2. Instrument classes (typical vendors/models & IO pattern), driver API (init/connect/configure/measure/abort/status), error codes, timeouts. 3. Experiment templates (parameter forms, ranges, sweep definitions, dwell times). 4. Raw data format (exact columns/arrays), QC checks, analysis pipeline, derived metrics, and report layout. 5. HIL simulator (stochastic models + ground-truth curves/images; noise/instrument artifacts). 6. Validation plan (golden samples, known materials, round-trip comparisons). 7. UI screens (wireframes): setup, live preview, progress, results, export. A) Electrical • 2-pt/4-pt probe: Sheet resistance (Van der Pauw), temp compensation; contact check. • Hall effect: Hall coefficient, μ, n/p; magnet sweep; sign detection. • I-V (diodes, BJTs, MOSFETs, solar cells): Compliance limits, sweep plans, stress avoidance; metrics (Is, n, β, gm, Vth, R_on, FF, η, Jsc, Voc, MPP). • C-V (MOS/Schottky): Small-signal AC superposed on DC, frequency sweeps, doping profiles, D_it, V_fb extraction. • DLTS/DLCP: Trap signatures, Arrhenius plots, capture cross-sections. • EBIC, PCD: Lifetime and recombination mapping on junctions. B) Optical • UV-Vis-NIR: Absorption/Transmission/Reflectance; Tauc plots; band-gap fits. • FTIR: Vibrational modes; baseline correction; peak fitting libraries. • Ellipsometry: Ψ/Δ modeling; multi-layer fits; thickness + n,k; library of dispersion models (Cauchy, Tauc-Lorentz). • PL/EL: Peak finding, FWHM, QE estimation; temperature-dependent PL; EL vs current. • Raman: Peak assignment, strain/stress extraction, phase ID. • Cathodoluminescence: Image + spectrum fusion; hyperspectral cubes. C) Structural/Morphological • XRD: θ–2θ; phase ID; Scherrer crystallite size; reciprocal space maps (optional). • SEM/TEM: Imaging pipelines; denoise/segmentation; scale calibration; EELS/AES hooks (if present). • AFM: Topography; roughness (Ra, Rq, Rsk, Rku); PSD analysis; tip deconvolution. • Profilometry: Step height, waviness; stitching long scans. D) Chemical/Elemental • XPS/XRF: Peak fitting, sensitivity factors, atomic %; charge referencing. • SIMS: Depth profiling; matrix effects; calibration with standards; MRP/TOF. • RBS: Layer thickness and areal density fitting; kinematic factor libraries. • NAA: Trace elemental workflows; decay corrections. • Chemical etch mapping: Pattern density vs etch rate maps. For every analysis pipeline, generate: algorithms, references, equations, default parameters, and validation datasets (synthetic + public). ⸻ 1. SPC, VM & ML • SPC: X-bar/R, EWMA, CUSUM; Cp/Cpk; alarms & triage; lot/wafer/device roll-ups; per-instrument control charts. • VM (Virtual Metrology): Template pipelines where equipment/FDC features + design/layout features → predicted metrology (e.g., thickness, roughness, band-gap). Provide training scripts, cross-validation, model cards, drift monitors, and ONNX export. • Anomaly/Drift: Unsupervised (IsolationForest, DBSCAN), supervised classification, and time-series forecasting (Prophet/ARIMA/LSTM, justify choice). • PM & Calibration Assist: Predictive maintenance suggestions, calibration reminders, calibration certificates store, uncertainty budgets. ⸻ 1. UX / UI • Home & Nav: Projects, Samples, Methods, Instruments, Runs, SPC, VM, Analytics, ELN. • Experiment Builder: Parameter forms, presets, safety checks, saved templates, batch scheduler. • Live Run: Real-time plots (I-V, spectra, thickness fits), wafer maps, camera feeds (if provided), run logs. • Results: Interactive plots (zoom/export), side-by-side comparisons, notebooks, “Generate Report” (PDF/HTML). • Imaging Suite: Galleries for SEM/TEM/AFM with ROIs, measurements, and annotation layers. • SPC Hub: Control charts, drill-down, “Explain Alert”, auto-root-cause suggestions. • VM Studio: Train/evaluate/deploy models; feature store; automated retraining. • Admin: Instruments registry, driver plugins, calibrations, roles, API keys, audit trails. Deliver Figma (or code-generated) wireframes, and a component library. ⸻ 1. Compliance, ELN/LIMS & Reproducibility • ELN: Rich text, images, plots, parameter snapshots, signatures. • LIMS: Sample lifecycle, custody, barcodes/QR, chain-of-custody logs. • Provenance: Every result links to instruments, versions, configs, raw data hashes. • Reports: Auto-generated PDF with summary, methods, parameters, plots, SPC status, approvals. • FAIR: Findable/Accessible/Interoperable/Reusable data policies; export packs (.zip) with README + schema. • SOP Library: Versioned SOPs; per-method pre-run checklists; printable forms. ⸻ 1. Security & Ops • RBAC (Admin/PI/Engineer/Technician/Viewer). • Audit logs and immutable run records. • Backups, retention, disaster recovery. • Secret management (Vault or SOPS). • Telemetry & observability (OpenTelemetry, Grafana, Loki/Prometheus). ⸻ 1. Testing & Validation • Unit tests for each parser, driver, and analysis function. • Golden datasets per method; expected metrics & tolerances. • End-to-end CI that spins containers, runs synthetic HIL sessions, checks SPC events, emits PDF reports, and verifies byte-exact artifacts. • Bench correlation plan: compare against vendor software for a subset of methods (when available). ⸻ 1. Documentation & Training • System Admin Guide, Lab User Guide, Method Playbooks (per method). • Embedded tooltips, hover cards, and “Explain this result” sidebars. • Tutorial projects: solar cell IV + EQE; MOS C-V; Raman + XRD phase ID; AFM roughness; XPS quant. ⸻ 1. Multi-Month Roadmap (Claude, generate the detailed plan) Produce a Gantt-like plan with Sessions (S1–S16 as needed): S1 – Program setup & architecture • PRD, risk register, compliance plan, repo scaffolding, container baseline, OpenAPI skeleton, UI shell. S2 – Data model & persistence • SQL migrations, ORMs, object storage scheme, unit tests, seed data. S3 – Instrument SDK & HIL • VISA/SCPI core, plugin SDK, 3 reference drivers (SMU, spectrometer, ellipsometer), HIL simulators. S4 – Electrical I (4PP, Hall) • Workflows, analysis, UI, reports, tests, synthetic datasets. S5 – Electrical II (IV, CV) • Diodes, MOSFETs, solar cells; safety; curve fitting; reports. S6 – Electrical III (DLTS/DLCP, EBIC/PCD) • Pipelines, visualization, validation. S7 – Optical I (UV-Vis-NIR, FTIR) • Spectra parsers, baseline/peak models, Tauc plots. S8 – Optical II (Ellipsometry, PL/EL, Raman) • Model fitting, dispersion libraries, peak ID, imaging overlays. S9 – Structural I (XRD) • Phase ID, crystallite size; RSM optional. S10 – Structural II (SEM/TEM/AFM/Profilometry) • Imaging pipelines, ROIs, roughness metrics; annotation tools. S11 – Chemical I (XPS/XRF) • Peak fitting/quantification; libraries and sensitivity factors. S12 – Chemical II (SIMS/RBS/NAA, chemical etch) • Depth profiling fits, kinematics, calibration flows. S13 – SPC Hub • Charts, alarms, triage, Cp/Cpk; alert explanations. S14 – VM & ML Suite • Feature store, training notebooks, model registry, inference service. S15 – LIMS/ELN & Reporting • ELN editor, SOPs, approvals, PDF engine; FAIR export. S16 – Hardening & Pilot • Performance profiling, security audit, HA deployment, pilot run report. Each session outputs code, tests, data, and a short PDF “Session Report” with screenshots. ⸻ 1. What to Produce Right Now (Session 1 baseline) 1. Program PRD (markdown): goals, stakeholders, success metrics, constraints. 1. Architecture doc with C4 diagrams. 1. Repo scaffold (monorepo or multi-repo with clear names), docker-compose.yml, helm/ skeletons. 1. OpenAPI starter (Auth, Instruments, Samples, Runs, Results). 1. DB migrations for core entities. 1. UI shell (Next.js) with nav + stub pages. 1. HIL sim framework and a trivial example (e.g., synthetic I-V of a diode). 1. Definition of Done checklist and acceptance tests for S1. Then proceed to S2 and beyond, following the roadmap. ⸻ 1. Quality Bars & Acceptance Criteria (apply every session) • All code builds in CI, unit tests ≥ 80% for core libs, integration tests for at least one end-to-end workflow. • Every method has: driver(s), simulator, workflow UI, analysis script, report, test data, and validation notes. • Documentation and examples updated; changelog entries; version bump. • Reproducible dev environment instructions verified on clean machine. ⸻ 1. Collaboration Aids • Generate issue lists and user stories per session. • Provide copy-paste CLI commands for developers. • Provide seed projects (e.g., “MOSFET lab”, “Solar cell line”, “GaN Raman+XRD”). • Produce glossary of terms and unit conventions. Proceed with the next session, follow every files I have you.

Files
28% of project capacity used
Retrieving

Session 6: Electrical III - Complete Delivery Package
264 lines

text



Session 6: Electrical III - Complete Documentation
575 lines

text



Session 6: Electrical III - Master Deployment Script
442 lines

text



Session 6: Complete Integration Tests
480 lines

text



Session 6: Electrical III - Complete Backend Analysis Modules
925 lines

text



Complete Session 6: Electrical III - UI Components
1,743 lines

text



Session 6: Electrical III - Complete Implementation Package
372 lines

text



Complete Integration Test Suite for Session 6: Electrical III
650 lines

text



Session 6: Electrical III - UI Components (Continued)
852 lines

text



Test Data Generators for Session 6: Electrical III (DLTS, EBIC, PCD)
717 lines

text



Session 6: Electrical III - DLTS, EBIC, PCD Implementation
1,629 lines

text



SemiconductorLab Platform - Session 5 Deployment Script
392 lines

text



Complete Integration Test Suite for Session 5: Electrical II
622 lines

text



SemiconductorLab Platform - Complete Implementation Roadmap & Status Report
398 lines

text



Complete Session 5: Electrical II - UI Components
970 lines

text



Session 5 Complete Delivery Package
486 lines

text



SemiconductorLab Platform - Master Deployment Script
520 lines

text



Test Session 5 Integration Test
632 lines

text



BJT Charachterization Interface
760 lines

text



C-V Profiling Interface
717 lines

text



MOSFET Characterization Interface
886 lines

text



Session 5 Method playbooks
1,190 lines

text



Complete Deployment Script
809 lines

text



Session 5 complete delivery package
541 lines

text



MOSFET Characterization
842 lines

text



C-V Profiling
824 lines

text



BJT characterization
696 lines

text



Complete integration test
675 lines

text



Session 5: Electrical II - Complete Implementation Guide
841 lines

text



Comprehensive Code Review & Production Roadmap
869 lines

text



MOSFET characterization UI
438 lines

text



Solar Cell Charachterization UI
476 lines

text



Session 5 Intégration test
637 lines

text



DLTS Analysis
177 lines

text



BJT I-V Analysis
603 lines

text



Session 5 test data generators
654 lines

text



Deployment and guide start
789 lines

text



Session 5 complete deliverables
651 lines

text



Semiconductor lab platform
375 lines

text



MOSFET I-V analysis
704 lines

text



Solar Cell I-V analysis
750 lines

text



C-V profiling analysis
749 lines

text



SemiconductorLab Platform - Complete Status Report
657 lines

text



MOSFET and solar Cell
746 lines

text



Capacitance-Voltage (C-V) Profiling Analysis
514 lines

text



Session 5 I-V and C-V test data
552 lines

text



Session 5 complete I-C I-V characterization
566 lines

text



Hall effects measurements UI
577 lines

text



Advanced Test Cases & Validation Scenarios
543 lines

text



Production deployment script
704 lines

text



Lab technician training guide
830 lines

text



I-V characterization
559 lines

text



Complete implementation summary
626 lines

text



4 point probe analysis
515 lines

text



Hall effects analysis module
551 lines

text



Electrical test data generators
484 lines

text



4 point probe
428 lines

text



Session 4 electrical 1 complete
580 lines

text



Ocean Optics Spectrometer
654 lines

text



S3 complete Test Suite
538 lines

text



S3 Complete Visa
517 lines

text



S3 Complete Plugin Architecture
490 lines

text



Keithley 2300
532 lines

text



Complete Implementation
658 lines

text



Complete Database
578 lines

text



SQL Alchemy
582 lines

text



Session 1 an 2 complete
705 lines

text



Docker Compose
314 lines

text



Make File
379 lines

text



Plugin Architectures
616 lines

text



Reference Drivers
656 lines

text



SQL Alchemy
644 lines

text



Pydantic schemas
674 lines

text



Object storage
658 lines

text



Unit handling system
544 lines

text



Test data generation
700 lines

text



Data model and specification
719 lines

text



Open API specification
1,188 lines

text



Database schema
603 lines

text



Next.js UI shell
384 lines

text



HL simulator Framework
470 lines

text



Session 1 Program setup and Architecture
571 lines

text



Semiconductor Lab platform
952 lines

text



Requirement Documents
497 lines

text



Architecture
965 lines

text



Semiconductor lab - Repository Structure
837 lines

text



VISA/SCPI
597 lines

text



Semiconductor lab - Repository Structure
27.25 KB •837 lines
•
Formatting may be inconsistent from source
# SemiconductorLab - Repository Structure

```
semiconductorlab/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Main CI pipeline
│   │   ├── release.yml               # Release automation
│   │   ├── security-scan.yml         # Dependency scanning
│   │   └── deploy-staging.yml        # Auto-deploy to staging
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── method_implementation.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── apps/
│   ├── web/                          # Next.js frontend
│   │   ├── app/                      # App router
│   │   │   ├── (auth)/
│   │   │   │   ├── login/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── register/
│   │   │   │       └── page.tsx
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx
│   │   │   ├── projects/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── samples/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── experiments/
│   │   │   │   ├── page.tsx          # Experiment list
│   │   │   │   ├── new/
│   │   │   │   │   └── page.tsx      # Experiment builder
│   │   │   │   └── [id]/
│   │   │   │       ├── page.tsx      # Experiment details
│   │   │   │       └── run/
│   │   │   │           └── page.tsx  # Live run viewer
│   │   │   ├── results/
│   │   │   │   ├── page.tsx
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx
│   │   │   ├── spc/
│   │   │   │   └── page.tsx          # SPC dashboard
│   │   │   ├── vm/
│   │   │   │   ├── page.tsx          # VM studio
│   │   │   │   ├── models/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── train/
│   │   │   │       └── page.tsx
│   │   │   ├── admin/
│   │   │   │   ├── instruments/
│   │   │   │   │   └── page.tsx
│   │   │   │   ├── users/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── calibrations/
│   │   │   │       └── page.tsx
│   │   │   ├── layout.tsx            # Root layout
│   │   │   └── page.tsx              # Home page
│   │   ├── components/
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── dialog.tsx
│   │   │   │   └── ...
│   │   │   ├── charts/
│   │   │   │   ├── IVCurvePlot.tsx
│   │   │   │   ├── SpectrumPlot.tsx
│   │   │   │   ├── WaferMap.tsx
│   │   │   │   └── ControlChart.tsx
│   │   │   ├── forms/
│   │   │   │   ├── ExperimentForm.tsx
│   │   │   │   ├── SampleForm.tsx
│   │   │   │   └── MethodSelector.tsx
│   │   │   ├── layouts/
│   │   │   │   ├── MainLayout.tsx
│   │   │   │   ├── DashboardLayout.tsx
│   │   │   │   └── Sidebar.tsx
│   │   │   └── shared/
│   │   │       ├── DataTable.tsx
│   │   │       ├── FileUpload.tsx
│   │   │       └── StatusBadge.tsx
│   │   ├── lib/
│   │   │   ├── api.ts                # API client
│   │   │   ├── auth.ts               # Auth helpers
│   │   │   ├── utils.ts              # Utilities
│   │   │   └── constants.ts
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── useExperiment.ts
│   │   │   └── useSSE.ts             # Server-Sent Events
│   │   ├── stores/
│   │   │   ├── authStore.ts          # Zustand store
│   │   │   └── experimentStore.ts
│   │   ├── types/
│   │   │   ├── api.ts                # API types
│   │   │   ├── experiment.ts
│   │   │   └── user.ts
│   │   ├── public/
│   │   │   ├── images/
│   │   │   └── icons/
│   │   ├── .env.local
│   │   ├── .env.example
│   │   ├── next.config.js
│   │   ├── tailwind.config.ts
│   │   ├── tsconfig.json
│   │   ├── package.json
│   │   └── README.md
│   │
│   └── docs/                         # Documentation site (optional)
│       └── ...
│
├── services/
│   ├── auth/                         # Authentication service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py               # FastAPI app
│   │   │   ├── config.py             # Settings
│   │   │   ├── database.py           # DB connection
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── user.py
│   │   │   │   └── session.py
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── user.py
│   │   │   │   └── token.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py
│   │   │   │   └── users.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth_service.py
│   │   │   │   └── user_service.py
│   │   │   └── dependencies.py       # FastAPI dependencies
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_auth.py
│   │   │   └── test_users.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   ├── pyproject.toml
│   │   └── README.md
│   │
│   ├── instruments/                  # Instrument service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument.py
│   │   │   │   ├── run.py
│   │   │   │   ├── calibration.py
│   │   │   │   └── measurement.py
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument.py
│   │   │   │   ├── run.py
│   │   │   │   └── measurement.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instruments.py
│   │   │   │   ├── runs.py
│   │   │   │   └── calibrations.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── instrument_service.py
│   │   │   │   ├── acquisition_service.py
│   │   │   │   └── calibration_service.py
│   │   │   ├── drivers/              # Instrument drivers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py           # Abstract base class
│   │   │   │   ├── connection.py     # Connection managers
│   │   │   │   ├── smu/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── keithley_2400.py
│   │   │   │   │   └── keithley_2600.py
│   │   │   │   ├── spectrometer/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── ocean_optics.py
│   │   │   │   │   └── avantes.py
│   │   │   │   ├── ellipsometer/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── ja_woollam.py
│   │   │   │   └── generic_scpi.py   # Fallback driver
│   │   │   ├── simulators/           # HIL simulators
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_simulator.py
│   │   │   │   ├── smu_simulator.py
│   │   │   │   ├── spectrometer_simulator.py
│   │   │   │   └── ellipsometer_simulator.py
│   │   │   ├── plugins/              # User-added drivers
│   │   │   │   └── README.md
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── visa_helper.py
│   │   │       └── data_validator.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_drivers.py
│   │   │   └── test_simulators.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── README.md
│   │
│   ├── analysis/                     # Analysis service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analysis_result.py
│   │   │   │   └── model.py          # ML models
│   │   │   ├── schemas/
│   │   │   │   ├── __init__.py
│   │   │   │   └── analysis.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── analysis.py
│   │   │   │   ├── spc.py
│   │   │   │   └── ml.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   └── analysis_service.py
│   │   │   ├── methods/              # Analysis methods
│   │   │   │   ├── __init__.py
│   │   │   │   ├── electrical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── four_point_probe.py
│   │   │   │   │   ├── hall_effect.py
│   │   │   │   │   ├── iv_analysis.py
│   │   │   │   │   ├── cv_analysis.py
│   │   │   │   │   └── dlts.py
│   │   │   │   ├── optical/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── uv_vis_nir.py
│   │   │   │   │   ├── ftir.py
│   │   │   │   │   ├── ellipsometry.py
│   │   │   │   │   ├── pl_el.py
│   │   │   │   │   └── raman.py
│   │   │   │   ├── structural/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── xrd.py
│   │   │   │   │   ├── sem_tem.py
│   │   │   │   │   └── afm.py
│   │   │   │   └── chemical/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── xps_xrf.py
│   │   │   │       ├── sims.py
│   │   │   │       └── rbs.py
│   │   │   ├── spc/                  # SPC algorithms
│   │   │   │   ├── __init__.py
│   │   │   │   ├── control_charts.py
│   │   │   │   ├── capability.py
│   │   │   │   └── rules.py
│   │   │   ├── ml/                   # ML models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── virtual_metrology.py
│   │   │   │   ├── anomaly_detection.py
│   │   │   │   └── drift_detection.py
│   │   │   ├── utils/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── curve_fitting.py
│   │   │   │   ├── statistics.py
│   │   │   │   └── units.py
│   │   │   └── workers/              # Celery tasks
│   │   │       ├── __init__.py
│   │   │       └── tasks.py
│   │   ├── tests/
│   │   │   ├── __init__.py
│   │   │   ├── conftest.py
│   │   │   ├── test_methods/
│   │   │   │   ├── test_iv_analysis.py
│   │   │   │   └── ...
│   │   │   ├── test_spc.py
│   │   │   └── test_ml.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── README.md
│   │
│   ├── reporting/                    # Report service
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   └── reports.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   └── report_service.py
│   │   │   ├── templates/            # Jinja2 templates
│   │   │   │   ├── base.html
│   │   │   │   ├── run_report.html
│   │   │   │   └── batch_report.html
│   │   │   └── generators/
│   │   │       ├── __init__.py
│   │   │       ├── pdf_generator.py
│   │   │       ├── csv_exporter.py
│   │   │       └── matlab_exporter.py
│   │   ├── tests/
│   │   │   └── test_generators.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   └── notifications/                # Notification service
│       ├── app/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── config.py
│       │   ├── routers/
│       │   │   ├── __init__.py
│       │   │   └── notifications.py
│       │   ├── services/
│       │   │   ├── __init__.py
│       │   │   └── notification_service.py
│       │   └── channels/
│       │       ├── __init__.py
│       │       ├── email.py
│       │       ├── slack.py
│       │       └── webhook.py
│       ├── tests/
│       │   └── test_channels.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
│
├── packages/                         # Shared libraries
│   ├── common/                       # Shared Python utilities
│   │   ├── semiconductorlab_common/
│   │   │   ├── __init__.py
│   │   │   ├── types.py              # Common types
│   │   │   ├── units.py              # Pint-based units
│   │   │   ├── constants.py
│   │   │   └── utils.py
│   │   ├── tests/
│   │   │   └── test_units.py
│   │   ├── setup.py
│   │   └── README.md
│   │
│   └── types/                        # Shared TypeScript types
│       ├── src/
│       │   ├── index.ts
│       │   ├── api.ts
│       │   ├── experiment.ts
│       │   └── user.ts
│       ├── tsconfig.json
│       ├── package.json
│       └── README.md
│
├── data/                             # Test data and seeds
│   ├── seeds/
│   │   ├── users.json
│   │   ├── instruments.json
│   │   ├── samples.json
│   │   └── runs.json
│   ├── test_data/
│   │   ├── electrical/
│   │   │   ├── iv_diode.csv
│   │   │   ├── hall_si.csv
│   │   │   └── ...
│   │   ├── optical/
│   │   │   ├── uv_vis_gan.csv
│   │   │   └── ...
│   │   └── ...
│   └── golden_samples/               # Reference datasets
│       ├── si_resistivity.json
│       ├── gaas_mobility.json
│       └── ...
│
├── db/                               # Database migrations
│   ├── migrations/
│   │   ├── 001_initial_schema.sql
│   │   ├── 002_add_calibrations.sql
│   │   └── ...
│   ├── alembic/                      # Alembic migrations (Python)
│   │   ├── versions/
│   │   │   └── ...
│   │   ├── env.py
│   │   └── alembic.ini
│   └── README.md
│
├── infra/                            # Infrastructure as Code
│   ├── docker/
│   │   ├── docker-compose.yml        # Development environment
│   │   ├── docker-compose.prod.yml   # Production-like local
│   │   └── .env.example
│   ├── kubernetes/
│   │   └── helm/
│   │       └── semiconductorlab/
│   │           ├── Chart.yaml
│   │           ├── values.yaml
│   │           ├── values-dev.yaml
│   │           ├── values-staging.yaml
│   │           ├── values-prod.yaml
│   │           └── templates/
│   │               ├── web-deployment.yaml
│   │               ├── instruments-deployment.yaml
│   │               ├── analysis-deployment.yaml
│   │               ├── postgres-statefulset.yaml
│   │               ├── redis-statefulset.yaml
│   │               ├── nats-statefulset.yaml
│   │               ├── ingress.yaml
│   │               ├── hpa.yaml
│   │               ├── pvc.yaml
│   │               ├── secrets.yaml
│   │               └── ...
│   ├── terraform/                    # Optional: for cloud infra
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── monitoring/
│       ├── prometheus/
│       │   └── prometheus.yml
│       ├── grafana/
│       │   ├── dashboards/
│       │   │   ├── system_overview.json
│       │   │   ├── instrument_metrics.json
│       │   │   └── spc_dashboard.json
│       │   └── datasources/
│       │       └── prometheus.yml
│       └── loki/
│           └── loki-config.yml
│
├── docs/                             # Documentation
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── c4_diagrams.md
│   │   └── security.md
│   ├── guides/
│   │   ├── admin_guide.md
│   │   ├── user_guide.md
│   │   └── developer_guide.md
│   ├── methods/                      # Method playbooks
│   │   ├── electrical/
│   │   │   ├── four_point_probe.md
│   │   │   ├── hall_effect.md
│   │   │   ├── iv_characterization.md
│   │   │   └── ...
│   │   ├── optical/
│   │   │   └── ...
│   │   └── ...
│   ├── api/
│   │   ├── openapi.yaml              # OpenAPI specification
│   │   └── examples/
│   │       └── curl_examples.sh
│   └── tutorials/
│       ├── getting_started.md
│       ├── first_experiment.md
│       └── advanced_analysis.md
│
├── scripts/                          # Utility scripts
│   ├── dev/
│   │   ├── setup.sh                  # Initial setup
│   │   ├── seed_db.py                # Seed database
│   │   └── generate_test_data.py     # Generate synthetic data
│   ├── ops/
│   │   ├── backup_db.sh              # Backup script
│   │   ├── restore_db.sh             # Restore script
│   │   └── health_check.sh
│   └── ci/
│       ├── run_tests.sh
│       └── deploy.sh
│
├── .gitignore
├── .dockerignore
├── .editorconfig
├── .prettierrc                       # Frontend formatting
├── .eslintrc.json                    # Frontend linting
├── pyproject.toml                    # Python project config (Ruff, Black)
├── Makefile                          # Common tasks
├── turbo.json                        # Turborepo config (if using Turborepo)
├── package.json                      # Root package.json (monorepo)
├── pnpm-workspace.yaml               # pnpm workspaces (or npm/yarn)
├── README.md                         # Project overview
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # Software license
└── CHANGELOG.md                      # Version history
```

-----

## Key Files Content

### Root `README.md`

```markdown
# SemiconductorLab Platform

> Enterprise-grade semiconductor characterization platform

[![CI](https://github.com/org/semiconductorlab/actions/workflows/ci.yml/badge.svg)](...)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Quick Start

### Prerequisites
- Docker 24+ & Docker Compose
- Node.js 20+ & pnpm 9+
- Python 3.11+
- Make

### Development Setup
```bash
# Clone repository
git clone https://github.com/org/semiconductorlab.git
cd semiconductorlab

# Start all services
make dev-up

# Access application
# Web: http://localhost:3000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001
```

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Admin Guide](docs/guides/admin_guide.md)
- [User Guide](docs/guides/user_guide.md)
- [API Reference](docs/api/openapi.yaml)

## Contributing

See <CONTRIBUTING.md>

## License

MIT License - see <LICENSE>

```
---

### `Makefile`
```makefile
.PHONY: help dev-up dev-down dev-logs test lint format build deploy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development
dev-up: ## Start development environment
	docker-compose -f infra/docker/docker-compose.yml up -d
	@echo "Services started. Web: http://localhost:3000"

dev-down: ## Stop development environment
	docker-compose -f infra/docker/docker-compose.yml down

dev-logs: ## Tail logs
	docker-compose -f infra/docker/docker-compose.yml logs -f

dev-reset: ## Reset database and start fresh
	docker-compose -f infra/docker/docker-compose.yml down -v
	make dev-up
	sleep 5
	make seed-db

# Database
migrate: ## Run database migrations
	cd services/instruments && alembic upgrade head

seed-db: ## Seed database with test data
	python scripts/dev/seed_db.py

# Testing
test: ## Run all tests
	./scripts/ci/run_tests.sh

test-unit: ## Run unit tests
	cd services/instruments && pytest tests/ -v
	cd services/analysis && pytest tests/ -v

test-e2e: ## Run end-to-end tests
	cd apps/web && pnpm test:e2e

# Linting & Formatting
lint: ## Lint all code
	cd apps/web && pnpm lint
	cd services/instruments && ruff check .
	cd services/analysis && ruff check .

format: ## Format all code
	cd apps/web && pnpm format
	cd services/instruments && ruff format .
	cd services/analysis && ruff format .

# Build
build: ## Build all Docker images
	docker-compose -f infra/docker/docker-compose.yml build

# Deployment (example - customize per environment)
deploy-staging: ## Deploy to staging
	helm upgrade --install semiconductorlab infra/kubernetes/helm/semiconductorlab \
		--namespace staging --create-namespace \
		--values infra/kubernetes/helm/semiconductorlab/values-staging.yaml

deploy-prod: ## Deploy to production (requires confirmation)
	@read -p "Deploy to PRODUCTION? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	helm upgrade --install semiconductorlab infra/kubernetes/helm/semiconductorlab \
		--namespace production --create-namespace \
		--values infra/kubernetes/helm/semiconductorlab/values-prod.yaml

# Cleanup
clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	cd apps/web && rm -rf .next node_modules
```

-----

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
dist/
*.egg-info/
.pytest_cache/

# Node
node_modules/
.next/
out/
.turbo/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Environment
.env
.env.local
.env.*.local
*.env

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Database
*.db
*.sqlite

# Logs
logs/
*.log

# Docker
.docker/

# Secrets
secrets/
*.pem
*.key
```

-----

### `pyproject.toml` (Root)

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "B", "C90"]
ignore = ["E501"]  # Line too long (handled by formatter)

[tool.black]
line-length = 120
target-version = ['py311']

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
pythonpath = ["services"]
```

-----

### `package.json` (Root)

```json
{
  "name": "semiconductorlab",
  "version": "1.0.0",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo run dev",
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "format": "prettier --write \"**/*.{ts,tsx,md,json}\""
  },
  "devDependencies": {
    "prettier": "^3.1.0",
    "turbo": "^1.11.0"
  },
  "engines": {
    "node": ">=20",
    "pnpm": ">=9"
  }
}
```

-----

### `turbo.json`

```json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": [".next/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {
      "dependsOn": ["^lint"]
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    }
  }
}
```

-----

## Setup Instructions

### 1. Clone and Initialize

```bash
git clone https://github.com/org/semiconductorlab.git
cd semiconductorlab

# Install frontend dependencies
pnpm install

# Create Python virtual environments (per service)
cd services/instruments
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Repeat for other services...
```

### 2. Configure Environment

```bash
# Copy example env files
cp apps/web/.env.example apps/web/.env.local
cp infra/docker/.env.example infra/docker/.env

# Edit .env files with your settings
```

### 3. Start Development Environment

```bash
make dev-up
```

### 4. Initialize Database

```bash
# Run migrations
make migrate

# Seed with test data
make seed-db
```

### 5. Verify Installation

```bash
# Check all services are running
docker ps

# Access services:
# Web UI: http://localhost:3000
# API Gateway: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Grafana: http://localhost:3001 (admin/admin)
```

-----

*End of Repository Scaffold*
