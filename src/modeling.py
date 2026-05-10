import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    save_metrics,
)


def split_patients(df, test_size=0.2, random_state=config.RANDOM_STATE):
    patients = pd.Series(df["patient_id"].unique()).sort_values().to_numpy()
    if len(patients) < 2:
        raise ValueError("Need at least two patients for a patient-level split")

    # Split by patient, not by window, to avoid leaking the same patient into both sets.
    patient_labels = (
        df.groupby("patient_id")["label"]
        .max()
        .reindex(patients)
        .astype(int)
        .to_numpy()
    )

    stratify = None
    values, counts = np.unique(patient_labels, return_counts=True)
    if len(values) == 2 and np.all(counts >= 2):
        stratify = patient_labels

    train_patients, test_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return set(train_patients), set(test_patients)


def make_xy(df):
    # The training script uses every feature column created by extract_features.py.
    drop_cols = ["patient_id", "window_start", "window_end", "label"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int)
    return x, y, feature_cols


def train_lightgbm_model(df, random_state=config.RANDOM_STATE):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install requirements with "
            "`pip install -r requirements.txt`."
        ) from exc

    train_patients, test_patients = split_patients(df, random_state=random_state)
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    x_train, y_train, feature_cols = make_xy(train_df)
    x_test, y_test, _ = make_xy(test_df)

    if y_train.nunique() < 2:
        raise ValueError("Training split has only one class; cannot train binary model")

    # Keep model settings fixed and simple for the first prototype.
    model = LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    # Evaluation is done on held-out patients only.
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    return model, metrics, y_test, y_pred, feature_cols


def save_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def train_and_save(feature_csv, model_path, metrics_path, figure_path, title):
    df = pd.read_csv(feature_csv)
    model, metrics, y_test, y_pred, _ = train_lightgbm_model(df)
    save_model(model, model_path)
    save_metrics(metrics, metrics_path)
    plot_confusion_matrix(y_test, y_pred, figure_path, title)
    return metrics
