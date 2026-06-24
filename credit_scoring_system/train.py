"""
train.py
Trains and evaluates multiple classifiers on the German Credit Dataset.
Models saved to models/ folder. Results printed and plotted.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed. Run: pip install xgboost  (will skip XGBoost model)")

from data_loader import load_data, preprocess

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid")


# ─── Cost-sensitive evaluation ─────────────────────────────────────────────────
# Per german.doc cost matrix: misclassifying Bad as Good costs 5x
def cost_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fn_cost = cm[1, 0] * 5   # Bad predicted as Good
    fp_cost = cm[0, 1] * 1   # Good predicted as Bad
    return fn_cost + fp_cost


def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cost = cost_score(y_test, y_pred)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                     "F1-Score": f1, "ROC-AUC": auc, "CostScore↓": cost}

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(f"  Cost     : {cost}  (lower is better; Bad→Good penalty x5)")
    print(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))
    return y_prob


def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for (name, fpr, tpr, auc_val), color in zip(roc_data, colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/05_roc_curves.png", dpi=150)
    plt.close()
    print("Saved: plots/05_roc_curves.png")


def plot_confusion_matrices(cm_data):
    n = len(cm_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_data):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
                    cbar=False, linewidths=0.5)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/06_confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: plots/06_confusion_matrices.png")


def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top20 = importances.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    top20.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title(f"Top 20 Feature Importances – {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/07_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: plots/07_feature_importance.png")


def plot_metrics_comparison(results):
    df = pd.DataFrame(results).T
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_plot = df[metrics].astype(float)
    x = np.arange(len(df_plot))
    width = 0.15
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df_plot[metric], width, label=metric, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison – All Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/08_model_comparison.png", dpi=150)
    plt.close()
    print("Saved: plots/08_model_comparison.png")


# ─── Main Training Pipeline ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CREDIT SCORING MODEL – GERMAN DATASET")
    print("=" * 60)

    # 1. Load & preprocess
    df = load_data()
    X, y = preprocess(df)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaler for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # 4. Define models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight={0: 1, 1: 5}, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10,
            class_weight={0: 1, 1: 5}, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight={0: 1, 1: 5}, n_jobs=-1, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=5, eval_metric="logloss",
            n_jobs=-1, random_state=42
        )

    results = {}
    roc_data = []
    cm_data  = []

    # 5. Train & evaluate each model
    for name, model in models.items():
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_prob = evaluate_model(name, model, X_te, y_test, results)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, roc_auc_score(y_test, y_prob)))
        cm_data.append((name, confusion_matrix(y_test, model.predict(X_te))))

        fname = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, fname)
        print(f"  → Saved: {fname}")

    # 6. Cross-validation on Random Forest
    print("\n" + "=" * 50)
    print("  5-Fold Cross-Validation (Random Forest)")
    print("=" * 50)
    rf = models["Random Forest"]
    cv = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc", n_jobs=-1)
    print(f"  AUC per fold : {np.round(cv, 4)}")
    print(f"  Mean AUC     : {cv.mean():.4f} ± {cv.std():.4f}")

    # 7. Plots
    plot_roc_curves(roc_data)
    plot_confusion_matrices(cm_data)
    plot_feature_importance(rf, X.columns, "Random Forest")
    plot_metrics_comparison(results)

    # 8. Summary
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    print(summary.to_string(float_format="{:.4f}".format))
    best = summary["ROC-AUC"].idxmax()
    print(f"\n  ✅ Best model by ROC-AUC: {best}  ({summary.loc[best, 'ROC-AUC']:.4f})")
    print("\nAll models saved to models/")
    print("All plots saved to plots/")


if __name__ == "__main__":
    main()
