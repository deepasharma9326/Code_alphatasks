"""
data_loader.py
Loads and preprocesses the German Credit Dataset.
"""

import pandas as pd
import numpy as np


# Column names based on german.doc attribute descriptions
COLUMN_NAMES = [
    "checking_account",       # A1  - qualitative
    "duration_months",        # A2  - numerical
    "credit_history",         # A3  - qualitative
    "purpose",                # A4  - qualitative
    "credit_amount",          # A5  - numerical
    "savings_account",        # A6  - qualitative
    "employment_since",       # A7  - qualitative
    "installment_rate",       # A8  - numerical (% of disposable income)
    "personal_status_sex",    # A9  - qualitative
    "other_debtors",          # A10 - qualitative
    "residence_since",        # A11 - numerical
    "property",               # A12 - qualitative
    "age",                    # A13 - numerical
    "other_installment_plans",# A14 - qualitative
    "housing",                # A15 - qualitative
    "existing_credits",       # A16 - numerical
    "job",                    # A17 - qualitative
    "dependents",             # A18 - numerical
    "telephone",              # A19 - qualitative
    "foreign_worker",         # A20 - qualitative
    "target"                  # 1=Good, 2=Bad
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment_since", "personal_status_sex", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker"
]

NUMERICAL_COLS = [
    "duration_months", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "dependents"
]


def load_data(filepath="data/german.data", verbose=True):
    """Load the raw German credit data file."""
    df = pd.read_csv(filepath, sep=" ", header=None, names=COLUMN_NAMES)
    # Convert target: 1=Good (0), 2=Bad (1) for binary classification
    df["target"] = df["target"].map({1: 0, 2: 1})
    if verbose:
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Good credit (0): {(df['target']==0).sum()}  |  Bad credit (1): {(df['target']==1).sum()}")
    return df


def preprocess(df, verbose=True):
    """
    Encode categorical features using one-hot encoding.
    Returns X (features), y (labels), and the encoded DataFrame.
    """
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    y = df_enc.pop("target")
    X = df_enc
    if verbose:
        print(f"After encoding: {X.shape[1]} features")
    return X, y


if __name__ == "__main__":
    df = load_data()
    print(df.head(3))
    X, y = preprocess(df)
    print(X.shape, y.value_counts())
