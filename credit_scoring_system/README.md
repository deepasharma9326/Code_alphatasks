# Credit Scoring System

A browser-based machine learning system for predicting applicant credit risk using the UCI Statlog German Credit dataset.

This improved version keeps your existing trained models and adds a proper web interface, prediction API, cleaner project structure, and VS Code run steps.

## Main Features

- Browser form for entering applicant details
- Model selection: Random Forest, Decision Tree, Logistic Regression
- Uses existing saved `.pkl` models from the `models/` folder
- Shows bad-credit probability and decision support message
- Includes training, EDA, plots, dataset, and prediction scripts

## Project Structure

```text
credit_scoring_system/
  app.py                  # Browser server and prediction API
  predict.py              # Loads saved model and predicts one applicant
  data_loader.py          # Loads and preprocesses German Credit dataset
  train.py                # Retrain models
  eda.py                  # Generate EDA plots
  requirements.txt        # App packages
  requirements-full.txt   # App + optional XGBoost training package
  data/                   # German credit dataset
  models/                 # Existing trained models
  plots/                  # Charts and reports
  static/                 # Browser UI files
```

## How To Run In VS Code

1. Open VS Code.

2. Go to `File > Open Folder`.

3. Select this folder:

```text
C:\Users\rajni\Documents\Codex\2026-06-19\i\credit_scoring_system
```

4. Open the VS Code terminal:

```text
Terminal > New Terminal
```

5. Install required packages once:

```powershell
pip install -r requirements.txt
```

6. Start the browser app:

```powershell
python app.py
```

7. Open this URL in your browser:

```text
http://127.0.0.1:8000
```

8. To stop the app, press `Ctrl + C` in the terminal.

## Retrain Models

The project already has trained models. To retrain the existing models:

```powershell
python train.py
```

To retrain with optional XGBoost support:

```powershell
pip install -r requirements-full.txt
python train.py
```

## Run Terminal Prediction Only

```powershell
python predict.py
```

## Dataset

- Source: UCI ML Repository, Statlog German Credit Data
- Rows: 1000 applicants
- Features: 20 applicant and loan attributes
- Target: Good credit or bad credit

## Important Note

This is an academic machine learning project. It should be used for demonstration and learning, not as a real bank approval system.
