# titanic_400
## Titanic Survival Prediction Project (Python + R)

This project predicts passenger survival from the Titanic dataset.
I used Python and R, and made Docker images for both so the grader can run everything easily.
The assignment goals: download data, clean it, train a model, and produce predictions.

Data is not included in this repo (required by the assignment).

```text
src/
 ├── app/               # Python code
 │   └── run_pipeline.py
 └── r/                 # R code + Dockerfile
     ├── pipeline.R
     ├── install_packages.R
     └── Dockerfile

models/                 # saved models (ignored)
outputs/                # predictions (ignored)
```

## Download Data (Required)

#### 1) Make sure Kaggle API key exists

Move kaggle.json to: ~/.kaggle/kaggle.json

Then run: chmod 600 ~/.kaggle/kaggle.json

#### 2) Download Titanic dataset

kaggle competitions download -c titanic -p src/data/
unzip -o src/data/*.zip -d src/data/

Files expected in src/data/:
```text
    train.csv
    test.csv
    gender_submission.csv
```

Run Python version
    Local
```text
    pip install -r requirements.txt
    python src/app/run_pipeline.py
    Creates:
    models/titanic_lr.joblib
    outputs/submission_python.csv

    Docker
    docker build -t titanic-python .
    docker run --rm \
    -v "$(pwd)/src/data:/app/src/data" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/outputs:/app/outputs" \
    titanic-python
```

```text
Run R version (Docker)
    docker build -f src/r/Dockerfile -t titanic-r .
    docker run --rm \
    -v "$(pwd)/src/data:/app/src/data" \
    -v "$(pwd)/outputs:/app/outputs" \
    titanic-r

    Creates:
    outputs/submission_r.csv
```
    
What the code does

#### Both pipelines:
	•	Read train.csv
	•	Add a simple feature (FamilySize)
	•	Fill missing values
	•	Train logistic regression
	•	Print training + validation accuracy
	•	Predict test.csv
	•	Save predictions in outputs/

Note: Kaggle does not give test labels, so true test accuracy cannot be calculated.

⸻

#### Notes for grader
	•	No dataset stored in repo
	•	Clear instructions to download data and run
	•	Both Python & R pipelines tested
	•	Docker works for both
	•	Print statements show each step in the pipeline

⸻

This finishes the project setup and execution as required.