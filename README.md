# Student Performance Prediction System

An end-to-end Python machine learning project that predicts **student pass/fail performance** using the UCI Student Performance dataset. The system includes data download, preprocessing, model training with hyperparameter tuning, evaluation with plots/reports, and a FastAPI inference API.

## Dataset
- **Source**: UCI Machine Learning Repository – Student Performance Data Set  
- **Download archive**: `https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`  
- This project uses `student-mat.csv` from that archive and saves it locally as:
  - `data/student_performance.csv`

## Project Structure

```text
.
├── data/
│   └── student_performance.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── api.py
│   └── utils.py
├── models/
│   └── best_model.pkl
├── tests/
│   └── smoke_test.py
├── plots/
│   └── roc_curve.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Run Data Download

```bash
python src/data_loader.py
```

This downloads from UCI and writes `data/student_performance.csv`.

## Run Model Training

```bash
python src/train_model.py
```

What it does:
- Builds a binary target (`pass=1`, `fail=0`) from `G3`.
- Applies preprocessing:
  - missing value imputation
  - one-hot encoding for categorical features
  - scaling for numerical features
- Trains and tunes:
  - Logistic Regression
  - Random Forest
  - SVM
  - Gradient Boosting
- Uses `GridSearchCV` (5-fold CV, scoring by F1)
- Saves the best estimator to `models/best_model.pkl`

## Run Evaluation

```bash
python src/evaluate.py
```

Outputs:
- Accuracy
- Precision / Recall / F1
- Confusion Matrix
- ROC curve plot (`plots/roc_curve.png`)
- JSON report (`evaluation_report.json`)

## Run the FastAPI Server

```bash
uvicorn src.api:app --reload
```

> If you get module import issues, run with:
> `PYTHONPATH=src uvicorn api:app --reload --app-dir src`

### API Endpoints
- `GET /` health check
- `POST /predict` inference endpoint

### Example `/predict` Input JSON

```json
{
  "school": "GP",
  "sex": "F",
  "age": 17,
  "address": "U",
  "famsize": "GT3",
  "Pstatus": "T",
  "Medu": 4,
  "Fedu": 4,
  "Mjob": "teacher",
  "Fjob": "services",
  "reason": "course",
  "guardian": "mother",
  "traveltime": 1,
  "studytime": 2,
  "failures": 0,
  "schoolsup": "no",
  "famsup": "yes",
  "paid": "no",
  "activities": "yes",
  "nursery": "yes",
  "higher": "yes",
  "internet": "yes",
  "romantic": "no",
  "famrel": 4,
  "freetime": 3,
  "goout": 3,
  "Dalc": 1,
  "Walc": 1,
  "health": 5,
  "absences": 2,
  "G1": 12,
  "G2": 13
}
```

### Example curl Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "school":"GP","sex":"F","age":17,"address":"U","famsize":"GT3",
    "Pstatus":"T","Medu":4,"Fedu":4,"Mjob":"teacher","Fjob":"services",
    "reason":"course","guardian":"mother","traveltime":1,"studytime":2,
    "failures":0,"schoolsup":"no","famsup":"yes","paid":"no",
    "activities":"yes","nursery":"yes","higher":"yes","internet":"yes",
    "romantic":"no","famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":1,
    "health":5,"absences":2,"G1":12,"G2":13
  }'
```

### Example `/predict` Output JSON

```json
{
  "prediction": 1,
  "label": "pass",
  "probability_pass": 0.92
}
```

## Tests / Validation

Run a lightweight smoke test to validate training artifact and prediction flow:

```bash
python tests/smoke_test.py
```

This script checks:
- dataset can be loaded,
- model artifact exists,
- prediction works on one example row.

## Notes on Features

The source data includes demographic, social, and academic attributes (e.g., parental education, support systems, prior grades, absences). This project predicts final outcome (`G3`) collapsed to pass/fail, which is more practical for intervention workflows.
