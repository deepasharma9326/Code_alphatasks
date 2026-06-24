import csv
import glob
import json
import math
import os
import random
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "work", "datasets"))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RANDOM_SEED = 42


HEART_FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_bp",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exercise_angina",
    "oldpeak",
    "slope",
    "major_vessels",
    "thal",
]

BREAST_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

DIABETES_FEATURES = [
    "avg_glucose",
    "max_glucose",
    "min_glucose",
    "glucose_readings",
    "regular_insulin",
    "nph_insulin",
    "ultralente_insulin",
    "meal_events",
    "exercise_events",
    "hypoglycemic_symptoms",
]


def read_heart():
    path = os.path.join(DATA_DIR, "heart", "processed.cleveland.data")
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) != 14 or "?" in parts:
                continue
            values = [float(part) for part in parts]
            rows.append((values[:13], 1 if values[13] > 0 else 0))
    return HEART_FEATURES, rows


def read_breast_cancer():
    path = os.path.join(DATA_DIR, "breast_cancer", "wdbc.data")
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            diagnosis = 1 if row[1] == "M" else 0
            rows.append(([float(value) for value in row[2:]], diagnosis))
    return BREAST_FEATURES, rows


def read_diabetes():
    folder = os.path.join(DATA_DIR, "diabetes", "Diabetes-Data")
    daily = defaultdict(lambda: defaultdict(float))
    glucose_codes = {48, 57, 58, 59, 60, 61, 62, 63, 64}
    for path in glob.glob(os.path.join(folder, "data-*")):
        patient_id = os.path.basename(path).split("-")[-1]
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    continue
                date, _time, code_text, value_text = parts
                try:
                    code = int(code_text)
                    value = float(value_text)
                except ValueError:
                    continue
                key = (patient_id, date)
                item = daily[key]
                if code in glucose_codes and value > 0:
                    item["glucose_sum"] += value
                    item["glucose_count"] += 1
                    item["max_glucose"] = max(item.get("max_glucose", value), value)
                    item["min_glucose"] = min(item.get("min_glucose", value), value)
                elif code == 33:
                    item["regular_insulin"] += value
                elif code == 34:
                    item["nph_insulin"] += value
                elif code == 35:
                    item["ultralente_insulin"] += value
                elif code in {66, 67, 68}:
                    item["meal_events"] += 1
                elif code in {69, 70, 71}:
                    item["exercise_events"] += 1
                elif code == 65:
                    item["hypoglycemic_symptoms"] += 1

    rows = []
    for item in daily.values():
        count = item.get("glucose_count", 0)
        if count == 0:
            continue
        avg_glucose = item["glucose_sum"] / count
        max_glucose = item.get("max_glucose", avg_glucose)
        min_glucose = item.get("min_glucose", avg_glucose)
        features = [
            avg_glucose,
            max_glucose,
            min_glucose,
            count,
            item.get("regular_insulin", 0),
            item.get("nph_insulin", 0),
            item.get("ultralente_insulin", 0),
            item.get("meal_events", 0),
            item.get("exercise_events", 0),
            item.get("hypoglycemic_symptoms", 0),
        ]
        label = 1 if max_glucose >= 200 or avg_glucose >= 180 else 0
        rows.append((features, label))
    return DIABETES_FEATURES, rows


def train_test_split(rows, ratio=0.2):
    rows = list(rows)
    random.Random(RANDOM_SEED).shuffle(rows)
    split_at = max(1, int(len(rows) * (1 - ratio)))
    return rows[:split_at], rows[split_at:]


def standardize(train_rows):
    columns = list(zip(*[features for features, _label in train_rows]))
    means = [sum(col) / len(col) for col in columns]
    scales = []
    for col, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in col) / len(col)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def train_logistic_regression(rows, epochs=2500, learning_rate=0.08, l2=0.001):
    train_rows, test_rows = train_test_split(rows)
    means, scales = standardize(train_rows)
    feature_count = len(train_rows[0][0])
    weights = [0.0] * feature_count
    bias = 0.0

    for _ in range(epochs):
        grad_w = [0.0] * feature_count
        grad_b = 0.0
        for raw_features, label in train_rows:
            features = [(value - mean) / scale for value, mean, scale in zip(raw_features, means, scales)]
            pred = sigmoid(bias + sum(w * x for w, x in zip(weights, features)))
            error = pred - label
            grad_b += error
            for i, value in enumerate(features):
                grad_w[i] += error * value
        size = len(train_rows)
        bias -= learning_rate * grad_b / size
        for i in range(feature_count):
            regularization = l2 * weights[i]
            weights[i] -= learning_rate * ((grad_w[i] / size) + regularization)

    return weights, bias, means, scales, evaluate(test_rows, weights, bias, means, scales)


def evaluate(rows, weights, bias, means, scales):
    if not rows:
        return {"accuracy": 0, "samples": 0}
    correct = 0
    for raw_features, label in rows:
        features = [(value - mean) / scale for value, mean, scale in zip(raw_features, means, scales)]
        pred = sigmoid(bias + sum(w * x for w, x in zip(weights, features)))
        correct += int((pred >= 0.5) == bool(label))
    return {"accuracy": round(correct / len(rows), 4), "samples": len(rows)}


def feature_defaults(rows, names):
    columns = list(zip(*[features for features, _label in rows]))
    defaults = {}
    for name, col in zip(names, columns):
        ordered = sorted(col)
        defaults[name] = round(ordered[len(ordered) // 2], 4)
    return defaults


def save_model(filename, title, names, rows, positive_label, negative_label, notes):
    weights, bias, means, scales, metrics = train_logistic_regression(rows)
    model = {
        "title": title,
        "features": names,
        "weights": weights,
        "bias": bias,
        "means": means,
        "scales": scales,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "metrics": metrics,
        "defaults": feature_defaults(rows, names),
        "notes": notes,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(model, file, indent=2)
    return path, metrics, len(rows)


def main():
    jobs = [
        (
            "heart.json",
            "Heart Disease Prediction",
            *read_heart(),
            "Heart disease risk detected",
            "No heart disease risk detected",
            "Trained from UCI processed Cleveland heart disease data. Target is disease presence: 0 vs 1-4.",
        ),
        (
            "breast_cancer.json",
            "Breast Cancer Diagnosis Prediction",
            *read_breast_cancer(),
            "Malignant tumor pattern",
            "Benign tumor pattern",
            "Trained from UCI Wisconsin Diagnostic Breast Cancer data. Target is M vs B.",
        ),
        (
            "diabetes.json",
            "Diabetes Glucose Risk Prediction",
            *read_diabetes(),
            "High glucose risk",
            "Lower glucose risk",
            "Trained from UCI Diabetes patient logs. This predicts high glucose risk, not diabetes diagnosis.",
        ),
    ]

    for filename, title, features, rows, positive, negative, notes in jobs:
        path, metrics, count = save_model(filename, title, features, rows, positive, negative, notes)
        print(f"{title}: saved {path} using {count} rows, test accuracy={metrics['accuracy']}")


if __name__ == "__main__":
    main()
