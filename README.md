# ⏳ Predicting Time to Failure  
**End-to-End Machine Learning | MLflow | Deployment-Ready**

This project predicts the **remaining time to failure (RUL)** of industrial machines using historical sensor and operational data.  
It demonstrates a complete **Machine Learning lifecycle** including preprocessing, feature engineering, model training, evaluation, experiment tracking with MLflow, and production-ready inference.

---

## 📌 Problem Statement

Unexpected machine failures cause costly downtime and maintenance issues.  
This system predicts **time-to-failure** to enable:

- Predictive maintenance  
- Reduced downtime  
- Optimized maintenance schedules  
- Improved asset reliability  

---

## 🚀 Key Features

- End-to-end ML pipeline  
- Regression-based RUL prediction  
- MLflow experiment tracking  
- Saved production model (`.pkl`)  
- Modular, scalable codebase  
- Docker-ready structure  

---

## 📂 Project Structure

```
predicting-time-to-failure/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── raw/
│       └── industrial_machine_time_to_failure.csv
│
├── mlruns/
│   └── 1/
│       └── models/
│
├── models/
│   └── rul_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── docker/
├── mlflow.db
├── requirements.txt
├── README.md
└── venv/
```

---

## 🧠 ML Pipeline

1. Data ingestion  
2. Data preprocessing  
3. Feature engineering  
4. Model training  
5. Model evaluation  
6. MLflow tracking  
7. Model serialization  
8. Inference pipeline  

---

## 🧪 MLflow Tracking

Launch MLflow UI:
```bash
mlflow ui
```

Access:
```
http://localhost:5000
```

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- MLflow  
- Joblib  

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/your-username/predicting-time-to-failure.git
cd predicting-time-to-failure
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python src/train_model.py
```

---

## 🔮 Predict RUL

```bash
python src/predict.py
```

---

## 🐳 Docker (Optional)

```bash
docker build -t time-to-failure .
docker run time-to-failure
```

---

## ✨ Author

**Swapnil Iwarkar**  
Machine Learning | Data Science | MLOps
