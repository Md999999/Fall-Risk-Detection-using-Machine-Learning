#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMU Synoptic Project: Live Clinical Demo
Researcher: Mosope Dada
Ethics Approval: 89091
"""

import sys
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

# Relative Path Fix
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kinecal_outputs" / "models" / "fall_risk_model.pkl"

if not MODEL_PATH.exists():
    print(f"Error: Model not found at {MODEL_PATH}.")
    print("Please run the training script first to save 'fall_risk_model.pkl'.")
    sys.exit()

# Load Model Data
print("Loading Trained AI Models...")
try:
    model_data = joblib.load(MODEL_PATH)
    models = model_data["models"]
    # FIXED: Matching the keys from your training script output
    feature_cols = model_data["cols"] 
    label_encoder = model_data["le"]   
except KeyError as e:
    print(f"KeyError: The model file is missing the expected key {e}.")
    print("Check that your training script uses the same keys in joblib.dump().")
    sys.exit()
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Webcam
cap = cv2.VideoCapture(0)
captured_frames = []

print("\n" + "="*40)
print("LIVE CLINICAL DEMO ACTIVE")
print("Instructions: Perform a sway task in view of the camera.")
print("Press 'q' to stop recording and run AI Analysis.")
print("="*40 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
        # Calculate Pelvis Center (Midpoint of Hips)
        px = (lm[23].x + lm[24].x) / 2
        py = (lm[23].y + lm[24].y) / 2
        pz = (lm[23].z + lm[24].z) / 2
        captured_frames.append(np.array([px, py, pz]))

    cv2.imshow('MMU Project: RGB Markerless Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# AI Inference Logic
if len(captured_frames) > 10:
    print("\nProcessing Biomechanical Features...")
    data_arr = np.array(captured_frames)
    
    # Extract Real-time Features
    range_x = np.ptp(data_arr[:, 0])
    # For a real-time demo, we assume a standard speed if movement is limited
    mean_speed = np.mean(np.linalg.norm(np.diff(data_arr[:, :2], axis=0), axis=1)) * 30 
    
    # Align features with training columns
    feats = {"range_x": range_x, "mean_speed": mean_speed} 
    X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    
    print("\n" + "*"*50)
    print("        CLINICAL ASSESSMENT RESULTS        ")
    print("*"*50)
    print(f" Total Frames Analyzed: {len(captured_frames)}")
    print(f" Measured Sway (Range X): {range_x:.4f}\n")
    
    for name, model in models.items():
        # Skip the baseline for the live UI output to keep it clean
        if name == "Logistic_Reg": continue
            
        raw_pred = model.predict(X)[0]
        label = label_encoder.inverse_transform([raw_pred])[0]
        print(f" [{name:15}] Predicted Risk: >> {label.upper()} <<")
        
    print("*"*50 + "\n")
else:
    print("\nRecording too short. Please capture more movement.")
