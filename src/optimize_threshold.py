
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data/raw/ai4i2020.csv")

TARGET = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[FEATURES]
y = df[TARGET]

# Même séparation stratifiée que précédemment
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=y_temp, random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )),
])

# 1. Entraînement uniquement sur le train
model.fit(X_train, y_train)

# 2. Recherche du seuil sur la validation
val_probabilities = model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# Le dernier point de precision/recall n'a pas de seuil associé
precision = precision[:-1]
recall = recall[:-1]

f2_scores = (
    5 * precision * recall
    / (4 * precision + recall + 1e-10)
)

best_index = np.argmax(f2_scores)
best_threshold = float(thresholds[best_index])

print("=== SEUIL OPTIMAL SUR VALIDATION ===")
print(f"Threshold : {best_threshold:.4f}")
print(f"Precision : {precision[best_index]:.4f}")
print(f"Recall : {recall[best_index]:.4f}")
print(f"F2-score : {f2_scores[best_index]:.4f}")

# 3. Réentraînement sur train + validation
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])

model.fit(X_train_full, y_train_full)

# 4. Évaluation finale sur le test jamais utilisé
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n=== RÉSULTATS FINAUX SUR TEST ===")
print("Precision:", precision_score(
    y_test, test_predictions, zero_division=0
))
print("Recall:", recall_score(
    y_test, test_predictions, zero_division=0
))
print("F1:", f1_score(
    y_test, test_predictions, zero_division=0
))
print("F2:", fbeta_score(
    y_test, test_predictions, beta=2, zero_division=0
))
print("Average Precision:", average_precision_score(
    y_test, test_probabilities
))

print("\n=== MATRICE DE CONFUSION ===")
print(confusion_matrix(y_test, test_predictions))

print("\n=== RAPPORT DE CLASSIFICATION ===")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Sans panne", "Avec panne"],
    zero_division=0,
))

# 5. Sauvegarde du modèle et du seuil
joblib.dump(
    model,
    ROOT / "models/predictive_maintenance_pipeline.joblib",
)

with open(ROOT / "models/decision_threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f, indent=4)

# 6. Graphique précision / recall
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label="Precision-Recall")
plt.scatter(
    recall[best_index],
    precision[best_index],
    label="Seuil sélectionné",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Optimisation du seuil de détection")
plt.legend()
plt.tight_layout()

plt.savefig(
    ROOT / "reports/figures/precision_recall_threshold.png",
    dpi=150,
)
plt.show()

print("\nModèle et seuil sauvegardés.")
