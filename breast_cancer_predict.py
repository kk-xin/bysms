#!/usr/bin/env python3
"""
Simple breast cancer prediction script using scikit-learn RandomForestClassifier.
Supports: train, evaluate, predict (from features or dataset sample index).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib
# use non-interactive backend for servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib


MODEL_DEFAULT = "model.joblib"


def train(model_path: str, n_estimators: int, test_size: float, random_state: int):
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained RandomForestClassifier — test accuracy: {acc:.4f}")
    # Save model and metadata
    out = {"model": clf, "feature_names": list(data.feature_names)}
    joblib.dump(out, model_path)
    print(f"Saved model to: {model_path}")


def evaluate(model_path: str):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    data = load_breast_cancer()
    saved = joblib.load(model_path)
    clf = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    X, y = data.data, data.target
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Evaluation on full dataset — accuracy: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=data.target_names))


def generate_report(model_path: str, out_dir: str = "reports"):
    outp = pathlib.Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    data = load_breast_cancer()
    saved = joblib.load(model_path)
    clf = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    X, y = data.data, data.target
    # Predictions
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy (full dataset): {acc:.4f}")
    print(classification_report(y, y_pred, target_names=data.target_names))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    conf_path = outp / "confusion_matrix.png"
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(conf_path)
    plt.close()
    print(f"Saved confusion matrix: {conf_path}")

    # ROC curve & AUC (binary)
    if len(np.unique(y)) == 2 and hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = outp / "roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved ROC curve: {roc_path}")
    else:
        print("ROC curve skipped: classifier lacks predict_proba or not binary.")

    # Feature importances for tree-based models
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]
        names = [data.feature_names[i] for i in idx]
        vals = importances[idx]
        plt.figure(figsize=(8, 8))
        sns.barplot(x=vals[:30], y=names[:30])
        plt.title("Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("")
        feat_path = outp / "feature_importances.png"
        plt.tight_layout()
        plt.savefig(feat_path)
        plt.close()
        print(f"Saved feature importances: {feat_path}")
    else:
        print("Feature importances not available for this model.")

    # Write a simple HTML report embedding the generated images and metrics
    try:
        html_path = outp / "report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>Breast Cancer Model Report</title></head><body>")
            f.write(f"<h1>Breast Cancer Model Report</h1>\n")
            f.write(f"<h2>Accuracy: {acc:.4f}</h2>\n")
            f.write("<h3>Classification report</h3>\n<pre>")
            f.write(classification_report(y, y_pred, target_names=data.target_names))
            f.write("</pre>\n")
            if conf_path.exists():
                f.write(f"<h3>Confusion Matrix</h3><img src=\"{conf_path.name}\" style=\"max-width:800px;\"><br/>\n")
            if (outp / "roc_curve.png").exists():
                f.write(f"<h3>ROC Curve</h3><img src=\"roc_curve.png\" style=\"max-width:800px;\"><br/>\n")
            if (outp / "feature_importances.png").exists():
                f.write(f"<h3>Feature Importances</h3><img src=\"feature_importances.png\" style=\"max-width:800px;\"><br/>\n")
            f.write("</body></html>")
        print(f"Saved HTML report: {html_path}")
    except Exception as e:
        print(f"Failed to write HTML report: {e}")


def predict(model_path: str, sample_index: int = None, features: str = None):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    data = load_breast_cancer()
    saved = joblib.load(model_path)
    if isinstance(saved, dict) and "model" in saved:
        clf = saved["model"]
        feature_names = saved.get("feature_names", list(data.feature_names))
    else:
        clf = saved
        feature_names = list(data.feature_names)

    if sample_index is not None:
        try:
            x = data.data[int(sample_index)]
        except IndexError:
            print(f"Sample index out of range. Must be 0..{len(data.data)-1}")
            sys.exit(1)
    elif features is not None:
        parts = [p.strip() for p in features.split(",") if p.strip()]
        if len(parts) != len(feature_names):
            print(f"Expected {len(feature_names)} features but got {len(parts)}")
            sys.exit(1)
        try:
            x = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            print("All feature values must be numeric.")
            sys.exit(1)
    else:
        print("Either --sample-index or --features must be provided for prediction.")
        sys.exit(1)

    pred = clf.predict([x])[0]
    pred_proba = clf.predict_proba([x])[0].tolist() if hasattr(clf, "predict_proba") else None
    out = {"prediction": int(pred), "prediction_name": data.target_names[int(pred)], "probability": pred_proba}
    print(json.dumps(out, indent=2, ensure_ascii=False))


def parse_args():
    p = argparse.ArgumentParser(description="Breast cancer RandomForest demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train a RandomForest on sklearn breast_cancer dataset")
    t.add_argument("--model-path", default=MODEL_DEFAULT)
    t.add_argument("--n-estimators", type=int, default=100)
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--random-state", type=int, default=42)

    e = sub.add_parser("evaluate", help="Evaluate a saved model on the full dataset")
    e.add_argument("--model-path", default=MODEL_DEFAULT)

    pr = sub.add_parser("predict", help="Predict using a saved model")
    pr.add_argument("--model-path", default=MODEL_DEFAULT)
    pr.add_argument("--sample-index", type=int, default=None)
    pr.add_argument("--features", type=str, default=None, help="Comma-separated feature values")
    r = sub.add_parser("report", help="Generate evaluation report and plots from a saved model")
    r.add_argument("--model-path", default=MODEL_DEFAULT)
    r.add_argument("--out-dir", default="reports", help="Output directory for plots")

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(args.model_path, args.n_estimators, args.test_size, args.random_state)
    elif args.cmd == "evaluate":
        evaluate(args.model_path)
    elif args.cmd == "predict":
        predict(args.model_path, sample_index=args.sample_index, features=args.features)
    elif args.cmd == "report":
        generate_report(args.model_path, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
