#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMU Synoptic Project: Human Motion Analysis Pipeline
Researcher: Mosope Dada
Ethics Approval Number: 89091
Description: This script performs feature extraction from the KINECAL dataset,
             trains a multi-model consensus (SVM, RF, XGBoost) plus a baseline, 
             and evaluates biomechanical fall risk.
"""

import matplotlib
matplotlib.use('Agg')
import os
import joblib
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# Ignore minor warnings for cleaner console output
warnings.filterwarnings("ignore")

@dataclass
class Config:
    # PORTABILITY FIX: Uses relative paths so the examiner can run the code
    base_script_path: Path = Path(__file__).resolve().parent
    
    # Folders located within the project directory
    base_dir: Path = base_script_path / "kinecal"
    output_dir: Path = base_script_path / "kinecal_outputs"
    
    fps: int = 30
    min_frames_required: int = 10
    random_state: int = 42
    n_splits: int = 5

CFG = Config()

def ensure_dirs():
    """Ensure output directory structure exists locally."""
    for d in ["figures", "reports", "models"]:
        (CFG.output_dir / d).mkdir(parents=True, exist_ok=True)

class Loader:
    def __init__(self, base_dir): 
        self.base = Path(base_dir)
        
    def load(self):
        """Loads .npy skeleton data from the KINECAL directory structure."""
        records = []
        if not self.base.exists():
            print(f"CRITICAL ERROR: Data folder not found at {self.base}")
            print("Please ensure the 'kinecal' folder is in the same directory as this script.")
            return records

        for subject in sorted(self.base.iterdir()):
            if not subject.is_dir() or not subject.name.isdigit(): 
                continue
            for task in sorted(subject.iterdir()):
                if not task.is_dir(): 
                    continue
                npys = sorted(task.glob("*.npy"))
                if not npys: 
                    continue
                
                arr = np.load(npys[0], allow_pickle=True)
                if arr.ndim == 3 and arr.shape[0] == 3: 
                    arr = np.transpose(arr, (2, 1, 0))
                
                if arr.shape[0] < CFG.min_frames_required: 
                    continue
                    
                records.append({
                    "subject_id": subject.name, 
                    "task": task.name, 
                    "skeleton": arr.astype(np.float32)
                })
        return records

class FeatureExtractor:
    def pelvis(self, skeleton): 
        return skeleton[:, 0, :2]

    def sway(self, traj):
        dist = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        return {
            "total_distance": float(np.sum(dist)), 
            "rms_sway": float(np.sqrt(np.mean(np.sum(traj ** 2, axis=1)))), 
            "mean_velocity": float(np.sum(dist) / max((len(traj) / CFG.fps), 1e-9)),
            "range_x": float(np.ptp(traj[:, 0]))
        }

    def temporal(self, traj):
        speed = np.linalg.norm(np.diff(traj, axis=0) * CFG.fps, axis=1)
        return {"mean_speed": float(np.mean(speed)) if len(speed) > 0 else 0.0}

    def extract(self, rec):
        traj = self.pelvis(rec["skeleton"])
        feats = {}
        feats.update(self.sway(traj))
        feats.update(self.temporal(traj))
        feats["subject_id"] = rec["subject_id"]
        feats["task"] = rec["task"]
        return feats

class LabelBuilder:
    def build(self, df):
        labels = []
        for _, row in df.iterrows():
            if row["total_distance"] > 0.3 or row["rms_sway"] > 0.1:
                labels.append("high")
            elif row["total_distance"] > 0.2:
                labels.append("moderate")
            else:
                labels.append("low")
        df["target"] = labels
        return df

class ModelComparison:
    def __init__(self):
        self.le = LabelEncoder()
        self.feature_cols = None
        self.models_dict = {
            "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=CFG.random_state),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=CFG.random_state, eval_metric='mlogloss'),
            "SVM": SVC(kernel='rbf', probability=True, random_state=CFG.random_state),
            "Baseline_LR": LogisticRegression(max_iter=1000, random_state=CFG.random_state)
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
            y_pred_all = np.zeros_like(y)
            for train_idx, test_idx in gkf.split(X, y, groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                y_pred_all[test_idx] = model.predict(X_test)
                
            acc = accuracy_score(y, y_pred_all)
            prec = precision_score(y, y_pred_all, average="weighted", zero_division=0)
            rec = recall_score(y, y_pred_all, average="weighted", zero_division=0)
            f1 = f1_score(y, y_pred_all, average="weighted", zero_division=0)
            
            results.append({
                "Model": model_name, 
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            })

            # Save visuals for main models
            if model_name != "Baseline_LR":
                cm = confusion_matrix(y, y_pred_all, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.le.classes_)
                disp.plot(cmap='Blues', values_format='.2f')
                plt.title(f"{model_name} Confusion Matrix")
                plt.savefig(CFG.output_dir / "figures" / f"cm_{model_name}.png", bbox_inches='tight')
                plt.close()

            model.fit(X, y)
            self.trained_models[model_name] = model

        df_all = pd.DataFrame(results)
        # Split baseline from main ranking table
        main_table = df_all[df_all["Model"] != "Baseline_LR"].copy()
        main_table["Rank (by F1)"] = main_table["F1-Score"].rank(ascending=False).astype(int)
        baseline_acc = df_all.loc[df_all["Model"] == "Baseline_LR", "Accuracy"].values[0]
        
        return main_table.sort_values("Rank (by F1)"), baseline_acc

    def save(self, path):
        joblib.dump({"models": self.trained_models, "le": self.le, "cols": self.feature_cols}, path)

def run():
    print(f"Initializing Analysis Pipeline (Ethics Approval: 89091)")
    ensure_dirs()
    
    records = Loader(CFG.base_dir).load()
    if not records: return

    df = LabelBuilder().build(pd.DataFrame([FeatureExtractor().extract(r) for r in records]))
    
    comparer = ModelComparison()
    results_df, baseline_acc = comparer.train_and_compare(df)
    
    # Save Model
    comparer.save(CFG.output_dir / "models" / "fall_risk_model.pkl")
    
    # --- FORMATTED OUTPUT START ---
    print("\n" + "="*80)
    print(" MACHINE LEARNING MODEL RANKING (CROSS-VALIDATED)")
    print("="*80)
    
    # Table Header
    header = f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Rank (by F1)'}"
    print(header)
    
    # Table Rows
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<15} {row['Accuracy']:<10.6f} {row['Precision']:<10.6f} {row['Recall']:<10.6f} {row['F1-Score']:<10.6f} {int(row['Rank (by F1)'])}")
    
    print("="*80 + "\n")
    
    # Summary Section
    xgb = results_df[results_df["Model"] == "XGBoost"].iloc[0]
    improvement = ((xgb['Accuracy'] - baseline_acc) / baseline_acc) * 100

    print(f"XGBoost Accuracy:   {xgb['Accuracy']:.4f}")
    print(f"XGBoost Precision:  {xgb['Precision']:.4f}")
    print(f"XGBoost Recall:     {xgb['Recall']:.4f}")
    print(f"XGBoost F1-Score:   {xgb['F1-Score']:.4f}")
    print(f"Baseline LR Acc:    {baseline_acc:.4f}")
    print(f"--> XGBoost outperformed baseline by {improvement:.2f}%")
    # --- FORMATTED OUTPUT END ---

if __name__ == "__main__":
    run()
