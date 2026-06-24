import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from model_utils import load_model, predict_label


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "heart": "heart.json",
    "breast_cancer": "breast_cancer.json",
    "diabetes": "diabetes.json",
}


def read_models():
    models = {}
    for key, filename in MODEL_FILES.items():
        models[key] = load_model(os.path.join(MODEL_DIR, filename))
    return models


class DiseasePredictionHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/metadata":
            models = read_models()
            payload = {
                key: {
                    "title": model["title"],
                    "features": model["features"],
                    "defaults": model["defaults"],
                    "metrics": model["metrics"],
                    "notes": model["notes"],
                    "positive_label": model["positive_label"],
                    "negative_label": model["negative_label"],
                }
                for key, model in models.items()
            }
            self.send_json(payload)
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json({"error": "Unknown endpoint"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model_key = payload["model"]
            values = payload["values"]
            model = read_models()[model_key]
            result = predict_label(model, values)
            result["model_title"] = model["title"]
            result["notes"] = model["notes"]
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DiseasePredictionHandler)
    print(f"Disease Prediction System running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
