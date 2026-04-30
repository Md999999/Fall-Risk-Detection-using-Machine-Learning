Here is the updated, professional README.md file with the UX Prototype section removed. This version is now perfectly streamlined for your technical submission.

Human Motion Analysis & Fall-Risk Assessment Pipeline
Researcher: Mosope Dada

Ethics Approval Number: 89091

Core Technology: Google MediaPipe (Markerless Motion Capture) & Multi-Model Machine Learning

1. Project Overview
This project aims to make clinical human motion analysis accessible and cost-effective. By replacing expensive optical tracking labs (e.g., Vicon) with a standard RGB webcam, this pipeline extracts 3D biomechanical metrics to objectively assess fall risk.

The system utilizes Google MediaPipe for joint tracking and a Multi-Model Consensus approach (SVM, Random Forest, and XGBoost) to classify patient risk based on sway, velocity, and temporal features.

2. Key Features
Hybrid Feature Engineering: Extraction of 18 biomechanical metrics (Sway Range, RMS Displacement, Mean Velocity).

Multi-Model Consensus: AI ensemble assessment using SVM, RF, and XGBoost.

Explainable AI: Integration of SHAP insights to visualize the drivers of risk classification.

GDPR-Compliant: Automated pseudonymisation of subject IDs for clinical safety.

Real-Time Assessment: Live clinical demo mode with digital skeleton overlay.

3. Prerequisites
OS: Windows 10 or 11.

Python Version: Python 3.11 ONLY.

Note: MediaPipe is currently incompatible with Python 3.12/3.13 binaries on Windows. Please ensure you are running 3.11 and have it added to your system PATH.

Hardware: Standard RGB Webcam for Live Demo.

4. Installation
Clone/Unzip the project folder to your machine.

Open Command Prompt and navigate to the root folder:

Bash
cd [Your_Folder_Path]
Install dependencies using the provided requirements file:

Bash
py -3.11 -m pip install -r requirements.txt


## 5. Project Structure
The code is designed to be **portable**. All paths are relative; as long as the internal folder structure is maintained, the scripts will run from any directory.

```text
.
├── Mosope_human_motion_analysis.py  # Phase 1: Pipeline training & evaluation
├── livedemo.py                      # Phase 2: Real-time clinical demo
├── requirements.txt                 # Project dependencies
├── kinecal/                         # Dataset directory (Subset included for testing)
├── kinecal_outputs/                 # Automatically generated artefacts
│   ├── models/                      # Saved trained models (fall_risk_model.pkl)
│   ├── figures/                     # Generated Confusion Matrices
│   └── reports/                     # Stats and feature tables
└── sample_output/                   # Pre-generated results for evaluation
6. How to Run
Phase 1: Training & Metrics
This script processes the dataset, extracts features, and trains the multi-model AI.

Bash
py -3.11 Mosope_human_motion_analysis.py
Output Artefacts:

Console: A ranking table comparing SVM, RF, and XGBoost performance.

Figures: Normalized Confusion Matrices saved in kinecal_outputs/figures/.

Model: A packaged fall_risk_model.pkl saved in kinecal_outputs/models/.

Phase 2: Live Clinical Demo
Run this once the model has been trained (or use the provided pre-trained model).

Bash
py -3.11 livedemo.py
Instructions:

Step back so the camera can see your full body (hips must be visible).

Ensure the digital skeleton is tracking your movement.

Perform a clinical movement (e.g., side-to-side sway) for 5 seconds.

Press 'q' to run the analysis. The terminal will display your biomechanical metrics and the risk prediction from all three AI models.

7. Dataset Information
The full KINECAL Dataset (86.2 GB) is hosted on PhysioNet.

Subset Included: A sample of 5 subjects is included in the /kinecal folder so the examiner can verify the pipeline without a large download.

Full Download: Instructions for wget or AWS CLI are included in the documentation if the full 86GB set is required.

8. Troubleshooting
ModuleNotFoundError (mediapipe): This usually happens if Windows defaults to Python 3.12. Always use the py -3.11 prefix. Ensure you haven't named any file mediapipe.py in your folder.

Webcam Initialization Error: Close other apps (Teams, Zoom) using the camera. Check "Camera Privacy Settings" in Windows to ensure desktop apps have access.

Path Error: If the script cannot find the data, ensure the kinecal folder is in the same directory as the script.

9. Ethics & License
Ethics Approval: 89091 (MMU Faculty of Science & Engineering).

This project is for academic use as part of a Final Year Dissertation. All data processed from the KINECAL set is pseudonymised to ensure patient privacy.
