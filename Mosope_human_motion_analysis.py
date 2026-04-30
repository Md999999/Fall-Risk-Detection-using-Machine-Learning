#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import os
import re
import json
import joblib
import warnings
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy import signal
from scipy.stats import entropy, f_oneway 

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

@dataclass
class Config:
    base_dir: str = r"C:\Users\mosop\KINECALWORK\kinecal"
    output_dir: str = r"C:\Users\mosop\KINECALWORK\kinecal_outputs"
    fps: int = 30
    min_frames_required: int = 10
    random_state: int = 42
    n_splits: int = 5
    target_mode: str = "biomechanical_risk" 
    clinical_label_csv: Optional[str] = None 
    clinical_subject_col: str = "subject_id"
    clinical_label_col: str = "label"
    drop_label_generating_features: bool = True

CFG = Config()

def ensure_dirs():
    for d in ["figures", "reports", "models"]:
        Path(os.path.join(CFG.output_dir, d)).mkdir(parents=True, exist_ok=True)

def normalize_subject_id(x): return str(x).strip()

def normalize_task_name(name):
    s = str(name).strip().lower()
    s = re.sub(r"^\d+_", "", s).replace("-", "_")
    return s

class Loader:
    def __init__(self, base_dir): self.base = Path(base_dir)
    def load(self):
        records = []
        for subject in sorted(self.base.iterdir()):
            if not subject.is_dir() or not subject.name.isdigit(): continue
            for task in sorted(subject.iterdir()):
                if not task.is_dir(): continue
                npys = sorted(task.glob("*.npy"))
                if not npys: continue
                arr = np.load(npys[0], allow_pickle=True)
                if arr.ndim == 3 and arr.shape[0] == 3: arr = np.transpose(arr, (2, 1, 0))
                if arr.shape[0] < CFG.min_frames_required: continue
                records.append({"subject_id": subject.name, "task": task.name, "skeleton": arr.astype(np.float32)})
        return records

class FeatureExtractor:
    def pelvis(self, skeleton): return skeleton[:, 0, :2]
    def sway(self, traj):
        dist = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        return {"total_distance": float(np.sum(dist)), "rms_sway": float(np.sqrt(np.mean(np.sum(traj ** 2, axis=1)))), 
                "mean_velocity": float(np.sum(dist) / max((len(traj) / CFG.fps), 1e-9)),
                "range_x": float(np.ptp(traj[:, 0]))}
    def temporal(self, traj):
        speed = np.linalg.norm(np.diff(traj, axis=0) * CFG.fps, axis=1)
        return {"mean_speed": float(np.mean(speed)) if len(speed) > 0 else 0.0}
    def extract(self, rec):
        traj = self.pelvis(rec["skeleton"])
        feats = {}
        feats.update(self.sway(traj)); feats.update(self.temporal(traj))
        feats["subject_id"] = rec["subject_id"]; feats["task"] = rec["task"]
        return feats

class LabelBuilder:
    def build(self, df):
        labels = []
        for _, row in df.iterrows():
            if row["total_distance"] > 0.3 or row["rms_sway"] > 0.1: labels.append("high")
            elif row["total_distance"] > 0.2: labels.append("moderate")
            else: labels.append("low")
        df["target"] = labels
        return df

class ModelComparison:
    def __init__(self):
        self.le = LabelEncoder()
        self.feature_cols = None
        # Defining the 3 core models
        self.models_dict = {
            "SVM": SVC(kernel='rbf', probability=True, random_state=CFG.random_state),
            "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=CFG.random_state),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=CFG.random_state)
        }
        self.trained_models = {}

    def get_features(self, df):
        X = df.drop(columns=["subject_id", "task", "target", "total_distance", "rms_sway", "mean_velocity"], errors="ignore").fillna(0.0)
        self.feature_cols = list(X.columns)
        return X

    def train_and_compare(self, df):
        X = self.get_features(df)
        y = self.le.fit_transform(df["target"].astype(str))
        groups = df["subject_id"].astype(str).values
        gkf = GroupKFold(n_splits=CFG.n_splits)
        
        results = []
        
        for model_name, model in self.models_dict.items():
            print(f"Training {model_name}...")
            y_pred_all = np.zeros_like(y)
            fold_acc = []
            
            # Cross Validation
            for train_idx, test_idx in gkf.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                y_pred_all[test_idx] = preds
                fold_acc.append(accuracy_score(y_test, preds))
                
            # Full metrics for this model
            acc = accuracy_score(y, y_pred_all)
            prec = precision_score(y, y_pred_all, average="weighted", zero_division=0)
            rec = recall_score(y, y_pred_all, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred_all, average="weighted", zero_division=0)
            
            results.append({"Model": model_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
            
            # Save Normalized Confusion Matrix Image
            cm = confusion_matrix(y, y_pred_all, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.le.classes_)
            disp.plot(cmap='Blues', values_format='.2f')
            plt.title(f"{model_name} Normalized Confusion Matrix")
            plt.savefig(os.path.join(CFG.output_dir, "figures", f"cm_{model_name}.png"), bbox_inches='tight')
            plt.close()

            # Train final model on ALL data for saving
            model.fit(X, y)
            self.trained_models[model_name] = model

        # Rank the results into a table
        df_results = pd.DataFrame(results)
        df_results["Rank (by F1)"] = df_results["F1-Score"].rank(ascending=False).astype(int)
        df_results = df_results.sort_values("Rank (by F1)")
        return df_results, X

    def save(self, path):
        joblib.dump({"models": self.trained_models, "label_encoder": self.le, "feature_cols": self.feature_cols}, path)

def run():
    ensure_dirs()
    loader = Loader(CFG.base_dir)
    records = loader.load()
    extractor = FeatureExtractor()
    df = pd.DataFrame([extractor.extract(r) for r in records])
    df = LabelBuilder().build(df)
    
    comparer = ModelComparison()
    results_df, X_full = comparer.train_and_compare(df)
    comparer.save(os.path.join(CFG.output_dir, "models", "fall_risk_model.pkl"))
    
    print("\n" + "="*50)
    print(" MACHINE LEARNING MODEL RANKING (CROSS-VALIDATED)")
    print("="*50)
    print(results_df.to_string(index=False))
    print("="*50)
    print(f"\nNormalized Confusion Matrices saved to {CFG.output_dir}/figures/")

if __name__ == "__main__":
    run()