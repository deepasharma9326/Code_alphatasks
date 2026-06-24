# Disease Prediction System

Browser-based academic project for Task 4: Disease Prediction from Medical Data.

The project trains and runs three classification models using UCI ML Repository datasets:

- Heart Disease: UCI processed Cleveland heart disease dataset.
- Breast Cancer: UCI Wisconsin Diagnostic Breast Cancer dataset.
- Diabetes: UCI diabetes patient logs, used here to predict high glucose risk because this archive contains records from diabetic patients rather than diabetes yes/no labels.

## Project Structure

```text
disease_prediction_system/
  app.py                 # Browser server and prediction API
  train_models.py        # Pure Python model training script
  model_utils.py         # Model loading and prediction helpers
  models/                # Trained JSON models
  static/                # Browser UI files
```

## How To Run In VS Code

1. Open this folder in VS Code:

```text
C:\Users\rajni\Documents\Codex\2026-06-19\i\disease_prediction_system
```

2. Open the VS Code terminal.

3. Run the app:

```powershell
python app.py
```

If your normal Python command does not work, use the bundled Python path:

```powershell
C:\Users\rajni\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe app.py
```

4. Open this URL in your browser:

```text
http://127.0.0.1:8000
```

## Retrain The Models

The trained models are already saved in `models/`. To retrain them after changing datasets or code:

```powershell
python train_models.py
```

## Algorithms Used

The models use logistic regression implemented in pure Python with feature standardization and a train/test split. This avoids extra package installation and makes the project easy to run on a college lab computer.

## Important Note

This system is for project demonstration and learning only. It is not a medical diagnosis tool.
