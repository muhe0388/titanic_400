import argparse
import subprocess
import sys

def run(cmd: list[str]):
    print("➡️ Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download")
    sub.add_parser("train")
    sub.add_parser("predict")

    args = parser.parse_args()

    if args.command == "download":
        run([sys.executable, "src/app/download_data.py"])
    elif args.command == "train":
        run([sys.executable, "src/app/train_model.py"])
    elif args.command == "predict":
        run([sys.executable, "src/app/predict.py"])

if __name__ == "__main__":
    main()