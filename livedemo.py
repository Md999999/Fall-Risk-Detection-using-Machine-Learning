#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MMU Synoptic Project: Live Clinical Demo
# Ethics Approval: 89091

import sys
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path

# Relative Path Fix - ensures the model is found relative to this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kinecal_outputs" / "models" / "fall_risk_model.pkl"

# Check if model exists before trying to load
if not MODEL_PATH.exists():
    print(f"Error: Model not found at {MODEL_PATH}.")
    print("Please run the 'Mosope_human_motion_analysis.py' script first to train and save the model.")
    sys.exit()

# Load Model Data
print("Loading Trained AI Models...")
try:
    model_data = joblib.load(MODEL_PATH)
    models = model_data["models"]
    feature_cols = model_data["feature_cols"]
    label_encoder = model_data["label_encoder"]
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
print("Instructions: Perform the task in view of the camera.")
print("Press 'q' to stop recording and run AI Analysis.")
print("="*40 + "\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break
        
    # Convert color for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Draw skeleton on screen for user feedback
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract Joint landmarks
        lm = results.pose_landmarks.landmark
        
        # Calculate Pelvis Center (Midpoint of Hips)
        # Joint 23: Left Hip, Joint 24: Right Hip
        px = (lm[23].x + lm[24].x) / 2
        py = (lm[23].y + lm[24].y) / 2
        pz = (lm[23].z + lm[24].z) / 2
        
        captured_frames.append(np.array([px, py, pz]))

    cv2.imshow('MMU Project: RGB Markerless Capture', frame)
    
    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

# Final classification logic
if len(captured_frames) > 10:
    print("\nProcessing Biomechanical Features...")
    
    # Convert to numpy for calculations
    data_arr = np.array(captured_frames)
    
    # Extract Demo Features (Range X / Mediolateral Sway)
    range_x = np.ptp(data_arr[:, 0])
    
    # Build a DataFrame for the models using the same feature columns as training
    # Note: Using placeholders for features not capturable via single joint demo
    feats = {"range_x": range_x, "mean_speed": 0.5} 
    X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
    
    print("\n" + "*"*50)
    print("        CLINICAL ASSESSMENT RESULTS        ")
    print("*"*50)
    print(f" Total Frames Analyzed: {len(captured_frames)}")
    print(f" Measured Sway (Range X): {range_x:.4f}\n")
    
    # Iterate through the dictionary of saved models (SVM, RF, XGBoost)
    for name, model in models.items():
        raw_pred = model.predict(X)[0]
        # Decode the numeric label (0, 1, 2) back to text (low, moderate, high)
        label = label_encoder.inverse_transform([raw_pred])[0]
        print(f" [{name:15}] Predicted Risk: >> {label.upper()} <<")
        
    print("*"*50 + "\n")
else:
    print("\nRecording too short. Please capture more movement for analysis.")
