
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data/raw/ai4i2020.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(y, predictions, zero_division=0),
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    # Les indicateurs de type de panne sont exclus
    # pour éviter la fuite de données.
    X = df[FEATURES]
    y = df[TARGET]

    # 70 % entraînement, 15 % validation, 15 % test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    print("=== DIMENSIONS DES ENSEMBLES ===")
    print(f"Entraînement : {len(X_train)}")
    print(f"Validation   : {len(X_val)}")
    print(f"Test         : {len(X_test)}")

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

    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}

    print("\n=== COMPARAISON SUR LA VALIDATION ===")

    for name, classifier in classifiers.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_val, y_val)

        results.append({"model": name, **metrics})
        fitted_models[name] = pipeline

        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by="f1", ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    print("\n=== CLASSEMENT DES MODÈLES ===")
    print(results_df.to_string(index=False))

    # Sélection du meilleur F1 sur la validation
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    print(f"\nModèle sélectionné : {best_name}")

    # Réentraînement sur entraînement + validation
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    best_model.fit(X_train_full, y_train_full)

    # Évaluation finale, une seule fois, sur le test
    test_metrics = evaluate_model(best_model, X_test, y_test)
    predictions = best_model.predict(X_test)

    print("\n=== RÉSULTATS FINAUX SUR LE TEST ===")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n=== MATRICE DE CONFUSION ===")
    print(confusion_matrix(y_test, predictions))

    print("\n=== RAPPORT DE CLASSIFICATION ===")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Sans panne", "Avec panne"],
        zero_division=0,
    ))

    joblib.dump(
        best_model,
        MODEL_DIR / "predictive_maintenance_pipeline.joblib",
    )

    with open(REPORT_DIR / "test_metrics.json", "w") as file:
        json.dump(
            {
                "selected_model": best_name,
                "test_metrics": test_metrics,
                "features": FEATURES,
            },
            file,
            indent=4,
        )

    print("\nModèle sauvegardé dans models/")
    print("Métriques sauvegardées dans reports/test_metrics.json")


if __name__ == "__main__":
    main()
