"""
eda.py
Exploratory Data Analysis for the German Credit Dataset.
Run this file to generate charts saved in the 'plots/' folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data, NUMERICAL_COLS

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].map({0: "Good", 1: "Bad"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
    ax.set_title("Credit Risk Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Credit Quality")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/01_target_distribution.png", dpi=150)
    plt.close()
    print("Saved: plots/01_target_distribution.png")


def plot_numerical_features(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUMERICAL_COLS):
        ax = axes[i]
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df["target"] == label][col]
            ax.hist(subset, bins=25, alpha=0.6, color=color,
                    label="Good" if label == 0 else "Bad", edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8)
    # hide unused subplot
    for j in range(len(NUMERICAL_COLS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Features by Credit Risk", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/02_numerical_features.png", dpi=150)
    plt.close()
    print("Saved: plots/02_numerical_features.png")


def plot_correlation_heatmap(df):
    num_df = df[NUMERICAL_COLS + ["target"]]
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = (corr.abs() < 0.05)  # hide very weak correlations for clarity
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation (Numerical)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: plots/03_correlation_heatmap.png")


def plot_credit_amount_by_purpose(df):
    purpose_map = {
        "A40": "New Car", "A41": "Used Car", "A42": "Furniture",
        "A43": "Radio/TV", "A44": "Appliances", "A45": "Repairs",
        "A46": "Education", "A48": "Retraining", "A49": "Business", "A410": "Others"
    }
    df2 = df.copy()
    df2["purpose_label"] = df2["purpose"].map(purpose_map).fillna("Other")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df2.groupby("purpose_label")["credit_amount"].median().sort_values(ascending=False).index
    sns.boxplot(data=df2, x="purpose_label", y="credit_amount", order=order,
                hue="target", palette={0: "#2ecc71", 1: "#e74c3c"}, ax=ax,
                legend=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_title("Credit Amount by Purpose & Risk", fontsize=13, fontweight="bold")
    ax.set_xlabel("Loan Purpose")
    ax.set_ylabel("Credit Amount (DM)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ecc71", label="Good"), Patch(color="#e74c3c", label="Bad")])
    plt.tight_layout()
    plt.savefig("plots/04_credit_amount_by_purpose.png", dpi=150)
    plt.close()
    print("Saved: plots/04_credit_amount_by_purpose.png")


if __name__ == "__main__":
    df = load_data()
    plot_target_distribution(df)
    plot_numerical_features(df)
    plot_correlation_heatmap(df)
    plot_credit_amount_by_purpose(df)
    print("\nAll EDA plots saved to plots/")
