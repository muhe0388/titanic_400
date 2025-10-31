import pandas as pd
from pathlib import Path
from joblib import load

DATA_DIR = Path("src/data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("models/titanic_model.joblib")

FEATURES = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

def main():
    print("Loading model...")
    model = load(MODEL_PATH)

    test = pd.read_csv(DATA_DIR / "test.csv")
    X_test = test[FEATURES]

    preds = model.predict(X_test)
    result = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds
    })

    OUT_PATH = OUT_DIR / "submission.csv"
    result.to_csv(OUT_PATH, index=False)

    print(f"Predictions saved to {OUT_PATH}")

if __name__ == "__main__":
    main()