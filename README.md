Here's the README rewritten to sound like a real person wrote it:


# KINECAL — Markerless Motion Pipeline & Live Demo
**Final Year Dissertation | Mosope Dada | Ethics Approval: 89091**


## What is this?

Clinical fall-risk assessment normally needs a £50k Vicon lab. This project does it with a webcam.

Using Google MediaPipe to track 3D joint movement in real time, the pipeline extracts 18 biomechanical features — things like sway range, RMS displacement, and mean velocity — then runs them through three ML models simultaneously (SVM, Random Forest, XGBoost). They vote. You get a risk classification. No markers, no lab, no nonsense.

There's also SHAP integration so you can actually *see* why the model made the call it did, and all subject IDs are pseudonymised automatically to keep things GDPR-compliant.

---

## Before you do anything — Python version matters

**Use Python 3.11. Not 3.12. Not 3.13. 3.11.**

MediaPipe has a C++ binary conflict with newer versions on Windows and it will just break silently. Make sure 3.11 is on your PATH, then install everything with:

```bash
py -3.11 -m pip install -r requirements.txt
```

You'll need Windows 10/11 and a standard webcam for the live demo.

---

## Project Structure

```
.
├── Mosope_human_motion_analysis.py  # Phase 1 — trains the models
├── livedemo.py                      # Phase 2 — live webcam demo
├── requirements.txt
├── kinecal/                         # Dataset folder (5-subject sample included)
├── kinecal_outputs/
│   ├── models/                      # fall_risk_model.pkl gets saved here
│   ├── figures/                     # Confusion matrices
│   └── reports/                     # Feature tables & stats
└── sample_output/                   # Pre-generated results if you just want to see outputs
```

Paths are all relative, so it doesn't matter where you unzip this — just keep the folder structure intact.

---

## Running it

### Phase 1 — Train the models

```bash
py -3.11 Mosope_human_motion_analysis.py
```

This loads the `.npy` arrays from `/kinecal`, extracts the biomechanical features, trains all three models with cross-validation, and prints a comparison table in the terminal. Confusion matrices go to `/figures`, and the trained model gets packaged into a `.pkl` file in `/models`.

### Phase 2 — Live demo

```bash
py -3.11 livedemo.py
```

- Step back far enough that the camera can see your full body — hips especially
- Wait for the skeleton overlay to lock on
- Do a clinical movement (side-to-side sway works well) for about 5 seconds
- Press `q` to stop
- The terminal will print your raw metrics and what each of the three models predicted

---

## Dataset
The KINECAL dataset is a large open-access biomechanics dataset hosted on PhysioNet.

The full KINECAL dataset is 86.2 GB. The pipeline processed 90 subjects and 453 movement trials in total. These were loaded directly from the local kinecal/ folder using the Loader class, which iterates every numbered subject directory and picks up all .npy skeleton arrays it finds. No manual filtering or subject selection was applied — the script just processes whatever is in the folder.

The full 86.2 GB dataset wasn’t downloaded in its entirety due to storage constraints, but the subset used is a genuine, unfiltered portion of the KINECAL data covering 90 subjects across multiple task categories. The pipeline is dataset-agnostic — point it at more data and it will process it identically.
. If you need the full set, `wget` and AWS CLI instructions are in the docs.

---

## Troubleshooting

**`module 'mediapipe' has no attribute 'solutions'`**
Almost certainly a Python version issue. Double-check you're running `py -3.11` and not just `python`. Also check you haven't accidentally named a file `mediapipe.py` in the project folder — Python will try to import that instead of the actual library.

**Webcam won't open**
Close Teams or Zoom first (they love to hold onto the camera). Then go to Windows Settings → Privacy → Camera and make sure desktop apps are allowed access.

**Path errors**
Make sure `kinecal/` is in the same directory as the scripts. That's it.

---

## Ethics

Ethics approval: **89091** (MMU Faculty of Science & Engineering). Academic use only. All KINECAL data is pseudonymised before processing.
