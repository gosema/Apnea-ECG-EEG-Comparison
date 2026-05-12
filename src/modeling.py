import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config
from src.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    save_metrics,
)


def make_xy(df):
    # The training script uses every feature column created by extract_features.py.
    drop_cols = ["patient_id", "window_start", "window_end", "label"]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"].astype(int)
    return x, y, feature_cols


def make_lightgbm_model(random_state=config.RANDOM_STATE):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install requirements with "
            "`pip install -r requirements.txt`."
        ) from exc

    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    )


def aggregate_fold_metrics(fold_metrics):
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mean = {}
    std = {}

    for metric_name in metric_names:
        values = [
            metrics[metric_name]
            for metrics in fold_metrics
            if metrics.get(metric_name) is not None
        ]
        if values:
            mean[metric_name] = float(np.mean(values))
            std[metric_name] = float(np.std(values))
        else:
            mean[metric_name] = None
            std[metric_name] = None

    return mean, std


def train_lightgbm_groupkfold(
    df,
    n_splits=5,
    random_state=config.RANDOM_STATE,
    modality_name=None,
):
    patients = pd.Series(df["patient_id"].unique()).sort_values().to_numpy()
    if len(patients) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} patients for {n_splits}-fold GroupKFold; "
            f"found {len(patients)}"
        )

    x, y, feature_cols = make_xy(df)
    groups = df["patient_id"]
    group_kfold = GroupKFold(n_splits=n_splits)

    folds = []
    oof_y_true = []
    oof_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        group_kfold.split(x, y, groups=groups),
        start=1,
    ):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

        if y_train.nunique() < 2:
            raise ValueError(
                f"Fold {fold_idx} training split has only one class; "
                "cannot train binary model"
            )

        model = make_lightgbm_model(random_state=random_state)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_val)
        y_proba = model.predict_proba(x_val)[:, 1]
        metrics = compute_classification_metrics(y_val, y_pred, y_proba)
        metrics = {metric: metrics.get(metric) for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]}

        fold_result = {"fold": fold_idx, **metrics}
        folds.append(fold_result)
        oof_y_true.append(y_val)
        oof_y_pred.append(pd.Series(y_pred, index=y_val.index))

        prefix = f"{modality_name} " if modality_name else ""
        print(f"{prefix}Fold {fold_idx}/{n_splits} metrics: {metrics}")

    mean, std = aggregate_fold_metrics(folds)
    cv_metrics = {
        "cv_type": "GroupKFold",
        "n_splits": n_splits,
        "folds": folds,
        "mean": mean,
        "std": std,
    }

    final_model = make_lightgbm_model(random_state=random_state)
    if y.nunique() < 2:
        raise ValueError("Full feature table has only one class; cannot train binary model")
    final_model.fit(x, y)

    return (
        final_model,
        cv_metrics,
        pd.concat(oof_y_true).sort_index(),
        pd.concat(oof_y_pred).sort_index(),
        feature_cols,
    )


def save_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def train_and_save(
    feature_csv,
    model_path,
    metrics_path,
    figure_path,
    title,
    modality_name=None,
):
    df = pd.read_csv(feature_csv)
    model, metrics, y_test, y_pred, _ = train_lightgbm_groupkfold(
        df,
        n_splits=5,
        modality_name=modality_name,
    )
    save_model(model, model_path)
    save_metrics(metrics, metrics_path)
    plot_confusion_matrix(y_test, y_pred, figure_path, title)
    return metrics
