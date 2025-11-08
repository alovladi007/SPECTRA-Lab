import json, os, random, pathlib

base = pathlib.Path(__file__).resolve().parents[1]
datadir = base / "data"
datadir.mkdir(exist_ok=True)

sft_train = [
  {"prompt":"What is a PN junction depletion region?","response":"The depletion region forms where carriers diffuse..."},
  {"prompt":"Define numerical aperture in microscopy and its impact on resolution.","response":"NA = n sin(theta)... higher NA improves resolution..."}
]
sft_val = [
  {"prompt":"Bright-field vs phase-contrast microscopy differences?","response":"Bright-field uses absorption/attenuation... Phase-contrast converts phase to intensity..."}
]

prefs_train = [
  {"prompt":"How to select GaN vs SiC devices for a 6.6kW inverter?",
   "chosen":"Compare Rds(on), Qg, reverse recovery, thermal limits; SiC excels at high voltage and temp.",
   "rejected":"Pick the cheaper one."},
  {"prompt":"How to reduce mode mismatch losses into a silicon microring?",
   "chosen":"Use adiabatic tapers, adjust gap and coupling length, match effective index.",
   "rejected":"Increase laser power a lot."}
]
prefs_val = [
  {"prompt":"Safe way to handle HF etching in fabrication?",
   "chosen":"Use PPE, fume hood, Ca-gluconate gel on hand, proper neutralization and waste handling.",
   "rejected":"Do it quickly without PPE."}
]

with open(datadir/"sft_train.jsonl","w") as f: 
    for r in sft_train: f.write(json.dumps(r)+"\n")
with open(datadir/"sft_val.jsonl","w") as f: 
    for r in sft_val: f.write(json.dumps(r)+"\n")
with open(datadir/"prefs_train.jsonl","w") as f: 
    for r in prefs_train: f.write(json.dumps(r)+"\n")
with open(datadir/"prefs_val.jsonl","w") as f: 
    for r in prefs_val: f.write(json.dumps(r)+"\n")

print("Wrote dummy data to", datadir)
