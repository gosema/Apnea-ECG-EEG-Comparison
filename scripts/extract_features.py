import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.annotations import load_apnea_events_for_patient
from src.data_loader import load_processed_directory
from src.features import build_feature_table


def _load_events(records):
    patient_ids = sorted({record["patient_id"] for record in records})
    return {
        patient_id: load_apnea_events_for_patient(patient_id)
        for patient_id in patient_ids
    }


def _extract_modality(modality, input_dir, output_path, default_fs):
    print(f"Loading processed {modality.upper()} files from {input_dir}")
    records = load_processed_directory(input_dir, modality=modality, default_fs=default_fs)
    print(f"Loaded {len(records)} {modality.upper()} records")

    if not records:
        print(f"Warning: no {modality.upper()} records found; skipping")
        return

    events_by_patient = _load_events(records)
    features = build_feature_table(records, events_by_patient, modality=modality)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(
        f"Saved {len(features)} {modality.upper()} windows "
        f"for {features['patient_id'].nunique()} patients to {output_path}"
    )


def main():
    _extract_modality(
        "ecg",
        config.ECG_PROCESSED_DIR,
        config.ECG_FEATURES_PATH,
        config.DEFAULT_ECG_FS,
    )
    _extract_modality(
        "eeg",
        config.EEG_PROCESSED_DIR,
        config.EEG_FEATURES_PATH,
        config.DEFAULT_EEG_FS,
    )


if __name__ == "__main__":
    main()
