# Apnea ECG EEG Comparison

Implementation of a digital signal processing pipeline using machine learning to compare sleep apnea detection performance between ECG and EEG signals using the MESA sleep dataset.

The project focuses on loading physiological sleep recordings, preprocessing ECG and EEG signals, extracting useful signal features, training independent models for each modality, and comparing their performance.

## Project structure

```text
Apnea-ECG-EEG-Comparison/
├── data/
│   ├── processed/                  # Processed data located at: https://drive.google.com/file/d/1qTkmaeSae_6qE2uwEEAG9s6Sec03TbW-/view?usp=sharing
│   └── mesa/                       # Local MESA dataset files, not tracked by Git
│
├── notebooks/                      # Exploratory analysis and experiments
│   ├── pickle_file_structure_extractio.ipynb    # Example about how data is stored in pickle files.
│   └── extraction_example.ipynb    # Examples of how the data is extracted from the edf files and how it's preprocessed.
│
├── src/                            # Reusable project code
│   ├── __init__.py
│   ├── config.py                   # Paths and general configuration
│   ├── data_loader.py              # Load EDF files, annotations, and metadata
│   ├── preprocessing.py            # Filter, normalize, and prepare EEG/ECG signals
│   ├── features.py                 # Extract signal features
│   ├── models.py                   # Define and train classifiers
│   ├── data_processor.py           # Loads data, applys filters, and obtains the fourier transform of the signals which is then stored into a file.
│   └── evaluation.py               # Metrics and model comparison
│
├── scripts/                        # Executable pipeline scripts
│   ├── extract_features.py         # Generate feature tables from signals
│   ├── train_lightgbm.py             # Train ECG and EEG models
│   ├── evaluate_models.py          # Evaluate and compare model results
│   └── info_edf.py                 # Extract from an EDF file the channels and frequency.
│
├── outputs/                        # Generated outputs, not tracked by Git
│   ├── features/
│   ├── models/
│   └── figures/
│
├── reports/                        # Report-related files and final figures
│   └── figures/
│
├── requirements.txt
├── channels_frequency.txt          # Channels and frequency used in the EDF
├── README.md
├── LICENSE
└── .gitignore
```

## Folder summary

- `data/`: local dataset directory. The MESA dataset should be placed under `data/mesa/`. Raw data should not be committed to Git.
- `notebooks/`: exploratory work, visual checks, and quick experiments.
- `src/`: reusable Python code. Files here should define functions/classes that can be imported elsewhere.
- `scripts/`: command-line scripts that run complete project steps using the code from `src/`.
- `outputs/`: generated files such as extracted features, trained models, and figures.
- `reports/`: figures or materials used in the final report or presentation.

## Basic setup

Create and activate a Python environment:

```bash
conda create -n pds python=3.11 -y
conda activate pds
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name pds --display-name "Python (pds)"
```

## Data location

The dataset must be placed locally inside:

```text
data/mesa/
```

Expected structure:

```text
data/
└── mesa/
    ├── actigraphy/
    ├── datasets/
    ├── documentation/
    ├── forms/
    ├── overlap/
    └── polysomnography/
```

## Current feature and model prototype

The current prototype works at 30-second window level. Apnea labels are created from the annotation event intervals: a window is positive if it overlaps a Hypopnea, Obstructive Apnea, Central Apnea, or Mixed Apnea event by at least 1 second.

ECG features are simple time-domain and R-peak based values computed per window:

```text
ecg_mean, ecg_std, ecg_rms, ecg_energy,
r_peak_count, heart_rate_mean, rr_mean, rr_std, rr_median, rmssd
```

EEG features are simple time-domain and spectral band-power values computed per window. The PSD is recomputed for each 30-second window, instead of using global PSD values from the pickle files:

```text
eeg_mean, eeg_std, eeg_rms, eeg_energy,
delta_power, theta_power, alpha_power, beta_power,
delta_relative, theta_relative, alpha_relative, beta_relative
```

Two separate LightGBM models are trained: one for ECG features and one for EEG features. The split is done at patient level, using 80% of patients for training and 20% for testing, so windows from the same patient are not shared across train and test sets.

Run the prototype from the repository root:

```bash
python scripts/extract_features.py
python scripts/train_lightgbm.py
```

Latest prototype results:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ECG LightGBM | 0.735 | 0.128 | 0.250 | 0.170 | 0.583 |
| EEG LightGBM | 0.819 | 0.118 | 0.105 | 0.111 | 0.607 |

Generated files are saved under `outputs/features/`, `outputs/models/`, `outputs/figures/`, and `outputs/metrics_*.json`.
