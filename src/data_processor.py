import data_loader
import preprocessing
import fourier
from pathlib import Path
import pickle

# We obtain in order the data using data_loader, filter the signals with preprocessing and obtain the fourier transform for both EEG and ECG.
def process_data(edf_path, xml_path, eeg_ch, ecg_ch):
    
    loader = data_loader.DataLoader()
    
    # Load with mne and normalize
    raw = loader.load_and_standardize(edf_path, [eeg_ch], [ecg_ch])

    # Apply band-pass filter with signal
    preprocessor = preprocessing.SignalPreprocessor()
    raw = preprocessor.apply_filters(raw)
    
    # Split the raw signal into EEG and ECG
    raw_eeg, raw_ecg = loader.split_raw(raw, [eeg_ch], [ecg_ch])

    # Obtain fourier transform for both EEG and ECG
    xf, psd = fourier.process_signal_fourier(raw_eeg, raw_eeg.info['sfreq'], type="EEG")
    lfhf = fourier.process_signal_fourier(raw_ecg, raw_ecg.info['sfreq'], type="ECG")
    
    # Cargamos las anotaciones desde el archivo XML y las superponemos a la señal temporal limpia
    raw = loader.load_annotations(raw, xml_path)

    return raw, xf, psd, lfhf

def save_results(xf, psd, lfhf, ecg_dir, eeg_dir, patient_id):
    # asegurar imports arriba: import numpy as np; import pickle
    ecg_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir.mkdir(parents=True, exist_ok=True)

    patient_id = edf_file.stem

    # comprobar y extraer canales (copias para no modificar `raw`)
    eeg_exist = [ch for ch in [EEG_channel] if ch in raw.ch_names]
    ecg_exist = [ch for ch in [ECG_channel] if ch in raw.ch_names]

    raw_eeg = raw.copy().pick_channels(eeg_exist) if eeg_exist else None 
    raw_ecg = raw.copy().pick_channels(ecg_exist) if ecg_exist else None
    
    if raw_eeg is None:
        print(f"Warning: No EEG channel found for {patient_id}.")
    if raw_ecg is None:
        print(f"Warning: No ECG channel found for {patient_id}.")

    # obtener arrays NumPy
    eeg_signal = raw_eeg.get_data() if raw_eeg is not None else None  # shape: (n_channels, n_samples)
    ecg_signal = raw_ecg.get_data()[0] if raw_ecg is not None else None  # primer canal ECG

    # Extraer anotaciones desde raw_eeg (que ya las tiene copiadas)
    annotations = None
    if raw_eeg is not None and raw_eeg.annotations is not None and len(raw_eeg.annotations) > 0:
        annotations = {
            'onset': raw_eeg.annotations.onset.tolist(),
            'duration': raw_eeg.annotations.duration.tolist(),
            'description': list(raw_eeg.annotations.description),
            'orig_time': raw_eeg.annotations.orig_time.isoformat() if raw_eeg.annotations.orig_time else None
        }
    
    # guardar ECG + lfhf (pickle preserva np.inf)
    with open(ecg_dir / f"{patient_id}_ecg.pkl", "wb") as f:
        pickle.dump({"ecg_signal": ecg_signal, "lfhf": lfhf, "annotations": annotations}, f)

    # guardar EEG + xf + psd
    with open(eeg_dir / f"{patient_id}_eeg.pkl", "wb") as f:
        pickle.dump({"eeg_signal": eeg_signal, "xf": xf, "psd": psd, "annotations": annotations}, f)

base_dir = Path(__file__).resolve().parent

repo_root = base_dir.parent

patient_dir = (repo_root / "data" / "mesa" / "polysomnography" / "edfs").resolve()
annotations_dir = (repo_root / "data" / "mesa" / "polysomnography" / "annotations-events-nsrr").resolve()
ecg_dir = (repo_root / "data" / "processed" / "ecg").resolve()
eeg_dir = (repo_root / "data" / "processed" / "eeg").resolve()

EEG_channel = "EEG3"
ECG_channel = "EKG"

for edf_file in patient_dir.glob("*.edf"):
    xml_file = annotations_dir / (edf_file.stem + "-nsrr.xml")
    if xml_file.exists() and edf_file.exists():
        raw, xf, psd, lfhf = process_data(edf_file, xml_file, EEG_channel, ECG_channel)
        save_results(xf, psd, lfhf, ecg_dir, eeg_dir, edf_file.stem)
    else:
        print(f"Error: No .EDF or .XML found for {edf_file.name}.")