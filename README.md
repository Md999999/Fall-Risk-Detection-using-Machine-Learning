KINECAL Markerless Motion Pipeline & Live Demo
Final Year Dissertation Project
Author: Mosope Dada


1. PROJECT OVERVIEW

This is the code repository for my final year dissertation. The goal of 
this project is to make clinical human motion analysis and fall-risk 
assessment much more accessible. Instead of relying on expensive optical 
tracking labs (like Vicon systems), this pipeline just needs a standard 
webcam. 

It uses Google MediaPipe to track 3D joint movements and feeds that 
data into three different machine learning models (SVM, Random Forest, 
and XGBoost). The AI then acts as a "Multi-Model Consensus" to classify 
a patient's fall risk based on objective metrics like sway and speed.

There are two main scripts you need to run:
1. kinecal_pipeline.py (Trains the AI)
2. livedemo.py (Tests it live using your webcam)


2. SYSTEM REQUIREMENTS

* OS: Windows 10 or 11
* Python: Python 3.11 ONLY. 

(Important note on Python: MediaPipe currently breaks on Python 3.12 
and 3.13 on Windows because of C++ binary issues. Make sure you are 
specifically using version 3.11 and have it added to your PATH).

To install everything you need, just open your Command Prompt and run:
py -3.11 -m pip install opencv-python mediapipe pandas xgboost scipy scikit-learn joblib matplotlib shap


3. FOLDER STRUCTURE

C:\Users\mosop\KINECALWORK\
│
├── kinecal_pipeline.py    # Main script (trains the AI and gets metrics)
├── livedemo.py            # The live webcam demo script
├── kinecal/               # Folder where the raw .npy dataset files go
└── kinecal_outputs/       # Where all the generated files get saved
    ├── models/            # Saves the trained models (fall_risk_model.pkl)
    ├── figures/           # Confusion matrix graphs get saved here
    └── reports/           # Stats and raw feature tables


4. HOW TO RUN PHASE 1: TRAINING

You have to train the machine learning models on the dataset before you 
can run the live demo.

1. Open Command Prompt.
2. Navigate to the project folder:
   cd C:\Users\mosop\KINECALWORK
3. Run the training script:
   py -3.11 kinecal_pipeline.py

When you run this, the script will:
* Load up the 3D `.npy` arrays from the `kinecal` folder.
* Calculate the biomechanical features.
* Train and cross-validate SVM, Random Forest, and XGBoost.
* Print out a ranking table in the terminal to show which model did best.
* Save the confusion matrices to the `figures` folder and package the 
  trained models into a `.pkl` file.


5. HOW TO RUN PHASE 2: LIVE DEMO

Once the models are trained and saved, you can test it live.

1. In your Command Prompt, make sure you're still in the project folder.
2. Run the demo script:
   py -3.11 livedemo.py

How to use it:
* Step back so the camera can see your whole body (especially your hips).
* You should see a digital skeleton tracking your movement on screen.
* Do a clinical movement (like swaying side-to-side or walking) for 
  about 3 to 5 seconds.
* Walk up and press the 'q' key to stop recording.
* Check the terminal: It will spit out your raw movement metrics and 
  show what all 3 AI models predicted your fall risk to be.


6. COMMON ERRORS & TROUBLESHOOTING
----------------------------------------------------------------------
* Error: "module 'mediapipe' has no attribute 'solutions'"
  Fix: This almost always means Windows is trying to use Python 3.12 or 
  3.13. Force it to use 3.11 by typing `py -3.11` instead of `python`. 
  Also, make sure you didn't accidentally name a file `mediapipe.py` 
  in your folder, or Python will try to import that instead of the 
  actual library.

* Error: Webcam doesn't open / CRITICAL ERROR
  Fix: Check your Windows Privacy Settings. Make sure "Camera access 
  for desktop apps" is toggled ON, otherwise Windows will block OpenCV 
  from using the lens.
======================================================================
