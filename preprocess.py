import wfdb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/"

def load_ecg_data(record_name):
    record = wfdb.rdrecord(os.path.join(DATA_PATH, record_name))
    annotation = wfdb.rdann(os.path.join(DATA_PATH, record_name), "atr")
    return record.p_signal, annotation.sample

def preprocess_signal(signal):
    scaler = StandardScaler()
    return scaler.fit_transform(signal)

if __name__ == "__main__":
    sample_record = "I01"  # Example file
    signal, labels = load_ecg_data(sample_record)
    processed_signal = preprocess_signal(signal)
    print(f"Processed shape: {processed_signal.shape}")
