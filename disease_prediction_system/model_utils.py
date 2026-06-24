import json
import math


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def load_model(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_probability(model, values):
    score = model["bias"]
    for name, weight, mean, scale in zip(
        model["features"], model["weights"], model["means"], model["scales"]
    ):
        raw_value = float(values[name])
        score += weight * ((raw_value - mean) / scale)
    return sigmoid(score)


def predict_label(model, values):
    probability = predict_probability(model, values)
    return {
        "probability": probability,
        "label": model["positive_label"] if probability >= 0.5 else model["negative_label"],
        "confidence": probability if probability >= 0.5 else 1 - probability,
    }
