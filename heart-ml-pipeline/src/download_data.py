import os
import io
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

# Known mirrors for a 303-row heart disease dataset (Cleveland-like schema).
CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/heart/heart.csv",
]

def try_download(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = io.StringIO(resp.text)
    df = pd.read_csv(buf)
    return df

def load_or_download() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    last_err = None
    for url in CANDIDATE_URLS:
        try:
            df = try_download(url)
            df.to_csv(CSV_PATH, index=False)
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download dataset to {CSV_PATH}. Last error: {last_err}")

if __name__ == "__main__":
    df = load_or_download()
    print(f"Dataset shape: {df.shape}; saved to {CSV_PATH}")
