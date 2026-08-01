# Hardware-compatible synthetic fNIRS dataset generator

This project generates matched healthy/hemorrhage synthetic samples for the supplied PCB:

- **16 LEDs total**: 8 at 740 nm and 8 at 850 nm
- **16 photodetectors**
- `I740.shape == (8, 16)`
- `I850.shape == (8, 16)`

The package stops at dataset generation. It does **not** contain MedT or any learning model.

## What is implemented

1. KiCad PCB parsing and exact hardware validation.
2. Fixed channel order and 256 source-detector channel map.
3. Flat layered phantom with a real air layer: air, scalp, skull, CSF, brain.
4. Matched healthy/hemorrhage anatomy from one subject.
5. Controlled hemorrhage depth (default 5–20 mm from the air/scalp boundary), size, and x-y sensor coverage.
6. Separate 740/850 optical-property tables with paired subject-level variation.
7. PMCX runner: 8 sources × 16 detectors per wavelength.
8. `sphere_mean` detector extraction, isolated behind a replaceable detector interface.
9. Optional noise (`none` or `basic`).
10. Atomic sample saving, resume, CSV metadata, logs, ETA, and pilot visualizations.

## Important scientific limitation

The current model is a **flat layered phantom**, not an anatomical human head. The detector measurement uses local PMCX flux averaged around the detector as an engineering approximation. These assumptions must be reported in the project and later validated against the physical phantom/hardware.

The included `analytic_debug` backend is only a software smoke test. It is **not Monte Carlo data and must never be used to train the model**.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install pmcx
```

PMCX requires a compatible NVIDIA CUDA environment. Verify it on the target computer:

```bash
python -c "import pmcx; print('PMCX imported successfully')"
```

## Validate PCB and tests

The supplied PCB is included as `fnirs.kicad_pcb`.

```bash
python run_geometry_validation.py --pcb fnirs.kicad_pcb --output output/geometry
```

Run all tests against the actual PCB:

**Windows PowerShell**

```powershell
$env:FNIRS_TEST_PCB="fnirs.kicad_pcb"
python -m pytest tests -v
```

**Linux/macOS**

```bash
FNIRS_TEST_PCB=fnirs.kicad_pcb python -m pytest tests -v
```

## First: software-only smoke test

This checks files, shapes, logs, resume, and plots. It is not physical data:

```bash
python run_optical_pilot.py --pcb fnirs.kicad_pcb --output output/debug_smoke --num-subjects 1 --backend analytic_debug --photons 1000 --noise-mode none
```

## Then: real PMCX pilot

Start with one pair and a low photon count:

```bash
python run_optical_pilot.py --pcb fnirs.kicad_pcb --output output/pmcx_pilot_1 --num-subjects 1 --backend pmcx --photons 100000 --noise-mode none
```

When this succeeds, inspect:

- `samples/*.npz`
- `metadata.csv`
- `generation.log`
- `visualizations/*_optical_pair.png`

Then generate 5–20 pilot pairs:

```bash
python run_optical_pilot.py --pcb fnirs.kicad_pcb --output output/pmcx_pilot_10 --num-subjects 10 --backend pmcx --photons 500000 --noise-mode none --resume
```

## Production generation

Only after the pilot has been reviewed:

```bash
python generate_optical_dataset.py --pcb fnirs.kicad_pcb --output data/fnirs_synthetic --num-subjects 1000 --backend pmcx --photons 2000000 --noise-mode basic --resume
```

Each subject creates two samples, one healthy and one hemorrhage. Thus `--num-subjects 1000` creates 2,000 files and performs 32 PMCX source simulations per subject.

## Runtime progress

The generator prints and writes to `generation.log`:

- subject index;
- healthy/hemorrhage case;
- wavelength;
- source index;
- completed PMCX runs;
- source runtime;
- estimated remaining time;
- pair runtime and optical difference.

A completed pair is saved before the next pair begins. `--resume` skips valid completed pairs.

## Sample contents

Every `.npz` includes:

- `I740`, `I850`;
- label, subject/pair/sample IDs;
- tissue volume, brain mask, 3D hemorrhage mask;
- hemorrhage center, radii, depth, and volume;
- all source and detector positions;
- 740/850 optical properties;
- seeds, photon count, versions, backend, and noise mode.

Load without pickle:

```python
import numpy as np
with np.load("sample.npz", allow_pickle=False) as d:
    print(d["I740"].shape, d["I850"].shape)
```

## Verified vs. not verified in this delivery

Verified in the provided environment:

- package imports;
- actual PCB parsing and 16/16 hardware validation;
- all unit tests;
- anatomy pairing and air layer;
- full end-to-end file generation using `analytic_debug`;
- output shapes, metadata, plots, logs, and atomic writes.

Not executable in the provided environment because PMCX/CUDA was unavailable:

- a real PMCX run and the exact runtime on the user's GPU;
- the installed PMCX version's exact returned field and numerical scale.

The PMCX integration follows the API used in the user's earlier scripts (`pmcx.run`, `flux`/`data`, summing the time dimension). Run the one-pair PMCX pilot before any large generation. If the installed PMCX version returns a different field or shape, the runner raises a clear error instead of silently writing invalid samples.
