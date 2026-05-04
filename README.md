# Apnea ECG EEG Comparison

Implementation of a digital signal processing pipeline using machine learning to compare sleep apnea detection performance between ECG and EEG signals using the MESA sleep dataset.

The project focuses on loading physiological sleep recordings, preprocessing ECG and EEG signals, extracting useful signal features, training independent models for each modality, and comparing their performance.

## Project structure

```text
Apnea-ECG-EEG-Comparison/
├── data/
│   └── mesa/                  # Local MESA dataset files, not tracked by Git
│
├── notebooks/                 # Exploratory analysis and experiments
│
├── src/                       # Reusable project code
│   ├── __init__.py
│   ├── config.py              # Paths and general configuration
│   ├── data_loader.py         # Load EDF files, annotations, and metadata
│   ├── preprocessing.py       # Filter, normalize, and prepare EEG/ECG signals
│   ├── features.py            # Extract signal features
│   ├── models.py              # Define and train classifiers
│   └── evaluation.py          # Metrics and model comparison
│
├── scripts/                   # Executable pipeline scripts
│   ├── extract_features.py    # Generate feature tables from signals
│   ├── train_models.py        # Train ECG and EEG models
│   ├── evaluate_models.py     # Evaluate and compare model results
│   └── info_edf.py            # Extract from an EDF file the channels and frequency.
│
├── outputs/                   # Generated outputs, not tracked by Git
│   ├── features/
│   ├── models/
│   └── figures/
│
├── reports/                   # Report-related files and final figures
│   └── figures/
│
├── requirements.txt
├── canales_frecuencia.txt     # Channels and frequency used in the EDF
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
