# src/app/run_pipeline.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump

DATA_DIR = Path("src/data")
OUT_DIR = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "titanic_lr.joblib"

FEATURES = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
TARGET = "Survived"

def main():
    print("=== Step 1: Load train.csv ===")
    train_path = DATA_DIR / "train.csv"
    df = pd.read_csv(train_path)
    print(f"[info] train shape={df.shape}")
    print("[preview] head:"); print(df.head(3))
    print("[info] missing counts:"); print(df[FEATURES+[TARGET]].isna().sum())

    print("\n=== Step 2: Simple feature adjustments (print changes) ===")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    feats = FEATURES + ["FamilySize"]
    print(f"[change] add feature FamilySize; FEATURES -> {feats}")

    print("\n=== Step 3: Build preprocessing + LogisticRegression ===")
    num_cols = ["Age","SibSp","Parch","Fare","Pclass","FamilySize"]
    cat_cols = ["Sex","Embarked"]
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, num_cols),
                             ("cat", categorical, cat_cols)])
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    print("[info] pipeline ready")

    print("\n=== Step 4: Split train/valid & train model (print accuracy) ===")
    X = df[feats].copy()
    y = df[TARGET].astype(int).copy()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[info] X_train={X_train.shape}, X_valid={X_valid.shape}")
    pipe.fit(X_train, y_train)
    y_pred_tr = pipe.predict(X_train)
    y_pred_va = pipe.predict(X_valid)
    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_va = accuracy_score(y_valid, y_pred_va)
    print(f"[metric] Train accuracy = {acc_tr:.4f}")
    print(f"[metric] Valid accuracy = {acc_va:.4f}")

    print("\n=== Step 5: Fit on full train & save model ===")
    pipe.fit(X, y)
    dump(pipe, MODEL_PATH)
    print(f"[artifact] Model saved to {MODEL_PATH}")

    print("\n=== Step 6: Load test.csv & predict ===")
    test = pd.read_csv(DATA_DIR / "test.csv")
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
    X_test = test[feats].copy()
    preds = pipe.predict(X_test).astype(int)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    out_csv = OUT_DIR / "submission_python.csv"
    sub.to_csv(out_csv, index=False)
    print(f"[artifact] predictions written to {out_csv}")

    print("\n=== Step 7: Test accuracy? (explanation) ===")
    print("The Kaggle test set has no ground-truth labels, so true test accuracy cannot be computed.")
    gender_path = DATA_DIR / "gender_submission.csv"
    if gender_path.exists():
        base = pd.read_csv(gender_path)[["PassengerId","Survived"]].rename(columns={"Survived":"Baseline"})
        joined = sub.merge(base, on="PassengerId", how="left")
        agree = np.mean(joined["Survived"] == joined["Baseline"])
        print(f"[proxy] Agreement with gender_submission = {agree:.4f} (NOT true accuracy)")

    print("\n=== DONE: All steps with verbose prints ===")

if __name__ == "__main__":
    main()