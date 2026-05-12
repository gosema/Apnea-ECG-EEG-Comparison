import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.modeling import train_and_save


def print_mean_metrics(modality, metrics):
    print(f"Mean {modality} metrics: {metrics['mean']}")
    print(f"Std {modality} metrics: {metrics['std']}")


def main():
    print("Training ECG model with 5-fold GroupKFold...")
    ecg_metrics = train_and_save(
        config.ECG_FEATURES_PATH,
        config.ECG_MODEL_PATH,
        config.ECG_METRICS_PATH,
        config.ECG_CONFUSION_MATRIX_PATH,
        "ECG LightGBM Confusion Matrix",
        modality_name="ECG",
    )
    print_mean_metrics("ECG", ecg_metrics)

    print("Training EEG model with 5-fold GroupKFold...")
    eeg_metrics = train_and_save(
        config.EEG_FEATURES_PATH,
        config.EEG_MODEL_PATH,
        config.EEG_METRICS_PATH,
        config.EEG_CONFUSION_MATRIX_PATH,
        "EEG LightGBM Confusion Matrix",
        modality_name="EEG",
    )
    print_mean_metrics("EEG", eeg_metrics)


if __name__ == "__main__":
    main()
