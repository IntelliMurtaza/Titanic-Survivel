import pandas as pd
import os
import joblib
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)

    