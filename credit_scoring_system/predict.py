"""
predict.py
Use a trained model to predict creditworthiness for a new applicant.
Run: python predict.py
"""

from pathlib import Path

import joblib
import pandas as pd
from data_loader import CATEGORICAL_COLS, load_data, preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "german.data"

MODEL_PATHS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    "decision_tree": MODELS_DIR / "decision_tree.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
}


def load_model(model_name="random_forest"):
    """Load a saved model."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_PATHS)}")
    return joblib.load(MODEL_PATHS[model_name])


def get_feature_columns():
    """Reconstruct the encoded feature column list used during training."""
    df = load_data(filepath=DATA_PATH, verbose=False)
    X, _ = preprocess(df, verbose=False)
    return X.columns.tolist()


def encode_applicant(applicant, feature_cols):
    """One-hot encode one applicant and align it to the training columns."""
    row = pd.DataFrame([applicant])
    row_enc = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=False)
    return row_enc.reindex(columns=feature_cols, fill_value=0)


def rebuild_logistic_regression():
    """Rebuild Logistic Regression for the installed scikit-learn version."""
    df = load_data(filepath=DATA_PATH, verbose=False)
    X, y = preprocess(df, verbose=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight={0: 1, 1: 5}, max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATHS["logistic_regression"])
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    return model, scaler


def predict_applicant(applicant: dict, model_name="random_forest") -> dict:
    """
    Predict credit risk for a single applicant.

    Returns 0 for good credit and 1 for bad credit.
    """
    feature_cols = get_feature_columns()
    row_enc = encode_applicant(applicant, feature_cols)
    model = load_model(model_name)
    rebuilt = False

    if model_name == "logistic_regression":
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        row_enc = scaler.transform(row_enc)

    try:
        pred = int(model.predict(row_enc)[0])
        prob_bad = float(model.predict_proba(row_enc)[0][1])
    except AttributeError as exc:
        if model_name != "logistic_regression" or "multi_class" not in str(exc):
            raise
        model, scaler = rebuild_logistic_regression()
        row_enc = scaler.transform(encode_applicant(applicant, feature_cols))
        pred = int(model.predict(row_enc)[0])
        prob_bad = float(model.predict_proba(row_enc)[0][1])
        rebuilt = True

    return {
        "prediction": pred,
        "probability_bad": round(prob_bad, 4),
        "probability_good": round(1 - prob_bad, 4),
        "risk_label": "BAD" if pred == 1 else "GOOD",
        "model_rebuilt": rebuilt,
    }


if __name__ == "__main__":
    example_applicant = {
        "checking_account": "A11",
        "duration_months": 24,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 3000,
        "savings_account": "A61",
        "employment_since": "A73",
        "installment_rate": 3,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since": 2,
        "property": "A121",
        "age": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "dependents": 1,
        "telephone": "A192",
        "foreign_worker": "A201",
    }

    result = predict_applicant(example_applicant, model_name="logistic_regression")
    print("\n" + "=" * 50)
    print("  CREDIT RISK PREDICTION")
    print("=" * 50)
    print(f"Prediction    : {result['risk_label']}")
    print(f"P(Bad Credit) : {result['probability_bad']:.1%}")
    if result["model_rebuilt"]:
        print("Logistic Regression model was rebuilt for this scikit-learn version.")
    print("=" * 50)
