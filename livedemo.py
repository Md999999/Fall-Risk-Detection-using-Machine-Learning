import sys
import cv2
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy import signal
from scipy.stats import entropy

warnings.filterwarnings("ignore")

class FeatureExtractor:
    def pelvis(self, skeleton): return skeleton[:, 0, :2]
    def sway(self, traj):
        diff = np.diff(traj, axis=0)
        dist = np.linalg.norm(diff, axis=1)
        return {
            "total_distance": float(np.sum(dist)), "rms_sway": float(np.sqrt(np.mean(np.sum(traj ** 2, axis=1)))), 
            "mean_velocity": float(np.sum(dist) / max((len(traj) / 30), 1e-9)),
            "range_x": float(np.ptp(traj[:, 0]))
        }
    def temporal(self, traj):
        speed = np.linalg.norm(np.diff(traj, axis=0) * 30, axis=1)
        return {"mean_speed": float(np.mean(speed)) if len(speed) > 0 else 0.0}
    def extract(self, rec):
        traj = self.pelvis(rec["skeleton"])
        feats = {}
        feats.update(self.sway(traj))
        feats.update(self.temporal(traj))
        return feats

print("Loading 3 Trained AI Models (SVM, Random Forest, XGBoost)...")
model_path = r"C:\Users\mosop\KINECALWORK\kinecal_outputs\models\fall_risk_model.pkl"

try:
    model_data = joblib.load(model_path)
    models = model_data["models"]  # This is now a dictionary of 3 models
    feature_cols = model_data["feature_cols"]
    label_encoder = model_data["label_encoder"]
    extractor = FeatureExtractor()
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    sys.exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): sys.exit()

print("\n" + "="*40)
print("LIVE CLINICAL DEMO STARTED (MULTI-MODEL)")
print("Press 'q' to stop recording and run AI.")
print("="*40 + "\n")

captured_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        
        pelvis_x = (lm[23].x + lm[24].x) / 2
        pelvis_y = (lm[23].y + lm[24].y) / 2
        pelvis_z = (lm[23].z + lm[24].z) / 2
        
        mapped_joints = np.zeros((10, 3))
        mapped_joints[0] = [pelvis_x, pelvis_y, pelvis_z]  
        captured_frames.append(mapped_joints)

    cv2.imshow('RGB Markerless Motion Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if len(captured_frames) > 10:
    print("\nProcessing Biomechanical Features...")
    skeleton_array = np.array(captured_frames, dtype=np.float32)
    mock_record = {"subject_id": "live_demo", "task": "assessment", "skeleton": skeleton_array}
    
    feats = extractor.extract(mock_record)
    df_feats = pd.DataFrame([feats])
    X = df_feats.reindex(columns=feature_cols, fill_value=0.0)
    
    print("\n" + "*"*50)
    print("        CLINICAL ASSESSMENT RESULT        ")
    print("*"*50)
    print(f" Frames Captured: {len(captured_frames)}")
    print(f" Mediolateral Sway (Range X): {feats['range_x']:.4f}\n")
    
    # Loop through all 3 models and print their predictions
    for model_name, model in models.items():
        raw_prediction = model.predict(X)[0]
        string_prediction = label_encoder.inverse_transform([raw_prediction])[0]
        print(f" [{model_name}] Predicted Risk: >> {str(string_prediction).upper()} <<")
        
    print("*"*50 + "\n")
else:
    print("\nNot enough movement recorded. Try again.")