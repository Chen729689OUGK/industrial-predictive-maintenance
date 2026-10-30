from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/ai4i2020.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}. "
            "Place le fichier CSV dans data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n=== APERÇU DU DATASET ===")
    print(df.head())

    print("\n=== DIMENSIONS ===")
    print(f"Lignes : {df.shape[0]}")
    print(f"Colonnes : {df.shape[1]}")

    print("\n=== TYPES DE VARIABLES ===")
    print(df.dtypes)

    print("\n=== VALEURS MANQUANTES ===")
    print(df.isnull().sum())

    print("\n=== RÉPARTITION DES DÉFAILLANCES ===")
    if "Machine failure" in df.columns:
        print(df["Machine failure"].value_counts(dropna=False))
        print(
            "\nPourcentages :\n",
            df["Machine failure"].value_counts(normalize=True) * 100,
        )
    else:
        print("La colonne 'Machine failure' est absente.")


if __name__ == "__main__":
    main()