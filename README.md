# F1 Race Prediction & Model Equivalence

This project compares two machine learning models—**Random Forest** and **Logistic Regression**—for predicting Formula 1 race outcomes. The goal is to evaluate whether the models behave equivalently across three layers: **performance**, **behavior**, and **structure**.

## 🚀 Setup

```
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## ▶️ Run Models & Tests

Train both models:
```
python f1_predictor.py
python f1_predictor_logistic.py
```

Performance equivalence:
```
python performance_equivalence_test.py
```

Behavioral equivalence:
```
python behavioral_equivalence_test.py
```

Structural equivalence:
```
python structural_equivalence_test.py
```

## 📊 Key Results

- **Performance Equivalence:**  
  McNemar’s Test → no significant difference (p = 0.58)

- **Behavioral Equivalence:**  
  Spearman correlation (RF vs LR) → **0.95**  
  Model vs real-world ranking → RF: **0.51**, LR: **0.62**

- **Structural Equivalence:**  
  Both models rely on similar key features.

## 📁 Main Files

```
f1_predictor.py
f1_predictor_logistic.py
data_loader.py
performance_equivalence_test.py
behavioral_equivalence_test.py
structural_equivalence_test.py
```


