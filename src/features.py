import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

from src import config
from src.annotations import label_windows


def window_times(signal_length, fs, window_seconds=config.WINDOW_SECONDS):
    # Use fixed non-overlapping windows so each feature row matches one label.
    samples_per_window = int(round(window_seconds * fs))
    if samples_per_window <= 0:
        raise ValueError("samples_per_window must be positive")

    n_windows = int(signal_length // samples_per_window)
    starts = np.arange(n_windows) * window_seconds
    ends = starts + window_seconds
    return starts, ends, samples_per_window


def _basic_features(window, prefix):
    return {
        f"{prefix}_mean": float(np.nanmean(window)),
        f"{prefix}_std": float(np.nanstd(window)),
        f"{prefix}_rms": float(np.sqrt(np.nanmean(window ** 2))),
        f"{prefix}_energy": float(np.nansum(window ** 2)),
    }


def _nan_r_peak_features():
    return {
        "r_peak_count": np.nan,
        "heart_rate_mean": np.nan,
        "rr_mean": np.nan,
        "rr_std": np.nan,
        "rr_median": np.nan,
        "rmssd": np.nan,
    }


def extract_ecg_features(window, fs):
    features = _basic_features(window, "ecg")

    clean = np.asarray(window, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < int(fs * 5) or np.nanstd(clean) == 0:
        features.update(_nan_r_peak_features())
        return features

    centered = clean - np.nanmedian(clean)
    prominence = max(0.5 * float(np.nanstd(centered)), 1e-6)
    min_distance = max(1, int(round(0.3 * fs)))

    # Detect simple R-peaks inside this window only; do not use global HRV values.
    peaks, _ = find_peaks(centered, distance=min_distance, prominence=prominence)
    if len(peaks) < 3:
        features.update(_nan_r_peak_features())
        return features

    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr >= 0.25) & (rr <= 2.5)]
    if rr.size < 2:
        features.update(_nan_r_peak_features())
        return features

    rmssd = np.nan
    if rr.size > 1:
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))

    features.update(
        {
            "r_peak_count": int(len(peaks)),
            "heart_rate_mean": float(np.mean(60.0 / rr)),
            "rr_mean": float(np.mean(rr)),
            "rr_std": float(np.std(rr)),
            "rr_median": float(np.median(rr)),
            "rmssd": rmssd,
        }
    )
    return features


def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_eeg_features(window, fs):
    features = _basic_features(window, "eeg")

    clean = np.asarray(window, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2 or np.nanstd(clean) == 0:
        powers = {
            "delta_power": np.nan,
            "theta_power": np.nan,
            "alpha_power": np.nan,
            "beta_power": np.nan,
            "delta_relative": np.nan,
            "theta_relative": np.nan,
            "alpha_relative": np.nan,
            "beta_relative": np.nan,
        }
        features.update(powers)
        return features

    nperseg = min(len(clean), int(round(fs * 4)))
    # PSD is computed per 30-second window so spectral features align with labels.
    freqs, psd = welch(clean, fs=fs, nperseg=nperseg)

    delta = _band_power(freqs, psd, 0.5, 4.0)
    theta = _band_power(freqs, psd, 4.0, 8.0)
    alpha = _band_power(freqs, psd, 8.0, 13.0)
    beta = _band_power(freqs, psd, 13.0, 30.0)
    total = _band_power(freqs, psd, 0.5, 30.0)

    if total and np.isfinite(total) and total > 0:
        rel = {
            "delta_relative": delta / total,
            "theta_relative": theta / total,
            "alpha_relative": alpha / total,
            "beta_relative": beta / total,
        }
    else:
        rel = {
            "delta_relative": np.nan,
            "theta_relative": np.nan,
            "alpha_relative": np.nan,
            "beta_relative": np.nan,
        }

    features.update(
        {
            "delta_power": delta,
            "theta_power": theta,
            "alpha_power": alpha,
            "beta_power": beta,
            **rel,
        }
    )
    return features


def build_feature_rows(record, events, modality):
    signal = record["signal"]
    fs = float(record["fs"])
    starts, ends, samples_per_window = window_times(len(signal), fs)
    labels = label_windows(starts, ends, events)

    rows = []
    extractor = extract_ecg_features if modality == "ecg" else extract_eeg_features

    for index, (start_time, end_time, label) in enumerate(zip(starts, ends, labels)):
        sample_start = index * samples_per_window
        sample_end = sample_start + samples_per_window
        window = signal[sample_start:sample_end]

        # Keep identifiers and label around the numeric features for later splitting.
        row = {
            "patient_id": record["patient_id"],
            "window_start": float(start_time),
            "window_end": float(end_time),
        }
        row.update(extractor(window, fs))
        row["label"] = int(label)
        rows.append(row)

    return rows


def build_feature_table(records, events_by_patient, modality):
    rows = []
    for record in records:
        events = events_by_patient.get(record["patient_id"], [])
        rows.extend(build_feature_rows(record, events, modality))
    return pd.DataFrame(rows)
