import os
import requests

def download_titanic_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data_dir = "src/data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "titanic.csv")

    print(f"Downloading Titanic dataset to {file_path} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with open(file_path, "wb") as f:
        f.write(resp.content)

    print("Download complete!")

if __name__ == "__main__":
    download_titanic_data()