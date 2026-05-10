import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.modeling import train_and_save


def main():
    print("Training ECG LightGBM model")
    ecg_metrics = train_and_save(
        config.ECG_FEATURES_PATH,
        config.ECG_MODEL_PATH,
        config.ECG_METRICS_PATH,
        config.ECG_CONFUSION_MATRIX_PATH,
        "ECG LightGBM Confusion Matrix",
    )
    print(f"ECG metrics: {ecg_metrics}")

    print("Training EEG LightGBM model")
    eeg_metrics = train_and_save(
        config.EEG_FEATURES_PATH,
        config.EEG_MODEL_PATH,
        config.EEG_METRICS_PATH,
        config.EEG_CONFUSION_MATRIX_PATH,
        "EEG LightGBM Confusion Matrix",
    )
    print(f"EEG metrics: {eeg_metrics}")


if __name__ == "__main__":
    main()
