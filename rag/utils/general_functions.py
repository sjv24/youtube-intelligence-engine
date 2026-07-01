import json
import pandas as pd
import os

# Base path - everything relative to repo root
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def getProcessedPath():
    return os.path.join(BASE_PATH, "data", "processed")

def getRawPath():
    return os.path.join(BASE_PATH, "data", "raw")

def getCSVPath():
    return os.path.join(BASE_PATH, "data")

def saveJson(data, filename):
    path = os.path.join(getProcessedPath(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {path}")

def readJson(filename):
    path = os.path.join(getProcessedPath(), filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def saveList(data, filename):
    path = os.path.join(getProcessedPath(), filename)
    df = pd.DataFrame(data, columns=["chunks"])
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

def getList(filename, column):
    path = os.path.join(getProcessedPath(), filename)
    df = pd.read_csv(path)
    return df[column].tolist()