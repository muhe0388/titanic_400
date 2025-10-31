import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_DIR = Path("src/data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "titanic_model.joblib"

FEATURES = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
TARGET = "Survived"

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

def build_model():
    numeric = ["Age","SibSp","Parch","Fare","Pclass"]
    categorical = ["Sex", "Embarked"]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ]
    )

    model = Pipeline([("preprocess", pre),
                      ("clf", LogisticRegression(max_iter=2000))])
    return model

def main():
    print("📦 Training model...")
    X, y = load_data()
    model = build_model()
    model.fit(X, y)
    dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
