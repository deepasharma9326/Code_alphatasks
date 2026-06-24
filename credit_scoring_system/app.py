"""
Browser app for the Credit Scoring System.

Run:
    python app.py
Then open:
    http://127.0.0.1:8000
"""

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from predict import predict_applicant


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_OPTIONS = {
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}

FIELD_GROUPS = [
    {
        "title": "Loan Details",
        "fields": [
            {
                "name": "checking_account",
                "label": "Checking account status",
                "type": "select",
                "options": {
                    "A11": "< 0 DM",
                    "A12": "0 to 200 DM",
                    "A13": ">= 200 DM",
                    "A14": "No checking account",
                },
            },
            {"name": "duration_months", "label": "Duration in months", "type": "number", "min": 4, "max": 72},
            {
                "name": "credit_history",
                "label": "Credit history",
                "type": "select",
                "options": {
                    "A30": "No previous credits",
                    "A31": "All credits paid back",
                    "A32": "Existing credits paid back",
                    "A33": "Delay in past payment",
                    "A34": "Critical account",
                },
            },
            {
                "name": "purpose",
                "label": "Loan purpose",
                "type": "select",
                "options": {
                    "A40": "New car",
                    "A41": "Used car",
                    "A42": "Furniture/equipment",
                    "A43": "Radio/TV",
                    "A44": "Domestic appliances",
                    "A45": "Repairs",
                    "A46": "Education",
                    "A48": "Retraining",
                    "A49": "Business",
                    "A410": "Other",
                },
            },
            {"name": "credit_amount", "label": "Credit amount", "type": "number", "min": 250, "max": 20000},
        ],
    },
    {
        "title": "Applicant Profile",
        "fields": [
            {
                "name": "savings_account",
                "label": "Savings account",
                "type": "select",
                "options": {
                    "A61": "< 100 DM",
                    "A62": "100 to 500 DM",
                    "A63": "500 to 1000 DM",
                    "A64": ">= 1000 DM",
                    "A65": "Unknown/no account",
                },
            },
            {
                "name": "employment_since",
                "label": "Employment since",
                "type": "select",
                "options": {
                    "A71": "Unemployed",
                    "A72": "< 1 year",
                    "A73": "1 to 4 years",
                    "A74": "4 to 7 years",
                    "A75": ">= 7 years",
                },
            },
            {"name": "installment_rate", "label": "Installment rate", "type": "number", "min": 1, "max": 4},
            {
                "name": "personal_status_sex",
                "label": "Personal status and sex",
                "type": "select",
                "options": {
                    "A91": "Male divorced/separated",
                    "A92": "Female divorced/separated/married",
                    "A93": "Male single",
                    "A94": "Male married/widowed",
                    "A95": "Female single",
                },
            },
            {"name": "age", "label": "Age", "type": "number", "min": 18, "max": 80},
        ],
    },
    {
        "title": "Financial Background",
        "fields": [
            {
                "name": "other_debtors",
                "label": "Other debtors",
                "type": "select",
                "options": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
            },
            {"name": "residence_since", "label": "Residence since", "type": "number", "min": 1, "max": 4},
            {
                "name": "property",
                "label": "Property",
                "type": "select",
                "options": {
                    "A121": "Real estate",
                    "A122": "Building society/life insurance",
                    "A123": "Car or other",
                    "A124": "Unknown/no property",
                },
            },
            {
                "name": "other_installment_plans",
                "label": "Other installment plans",
                "type": "select",
                "options": {"A141": "Bank", "A142": "Stores", "A143": "None"},
            },
            {
                "name": "housing",
                "label": "Housing",
                "type": "select",
                "options": {"A151": "Rent", "A152": "Own", "A153": "Free"},
            },
        ],
    },
    {
        "title": "Work And Contact",
        "fields": [
            {"name": "existing_credits", "label": "Existing credits", "type": "number", "min": 1, "max": 4},
            {
                "name": "job",
                "label": "Job",
                "type": "select",
                "options": {
                    "A171": "Unemployed/unskilled non-resident",
                    "A172": "Unskilled resident",
                    "A173": "Skilled employee",
                    "A174": "Management/self-employed",
                },
            },
            {"name": "dependents", "label": "Dependents", "type": "number", "min": 1, "max": 2},
            {
                "name": "telephone",
                "label": "Telephone",
                "type": "select",
                "options": {"A191": "No", "A192": "Yes"},
            },
            {
                "name": "foreign_worker",
                "label": "Foreign worker",
                "type": "select",
                "options": {"A201": "Yes", "A202": "No"},
            },
        ],
    },
]

DEFAULT_APPLICANT = {
    "checking_account": "A12",
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


class CreditScoringHandler(SimpleHTTPRequestHandler):
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
            self.send_json(
                {
                    "models": MODEL_OPTIONS,
                    "field_groups": FIELD_GROUPS,
                    "defaults": DEFAULT_APPLICANT,
                }
            )
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json({"error": "Unknown endpoint"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            model_name = payload.get("model_name", "random_forest")
            applicant = payload["applicant"]
            result = predict_applicant(applicant, model_name=model_name)
            result["model_name"] = MODEL_OPTIONS.get(model_name, model_name)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CreditScoringHandler)
    print(f"Credit Scoring System running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
