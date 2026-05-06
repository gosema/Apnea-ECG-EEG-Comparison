import mne
import sys
import os

# Configure the path so Python can find the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our classes
from src.data_loader import DataLoader
from src.preprocessing import SignalPreprocessor

def main():
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    paciente_id = "0006"  # TODO: test with patient 6, but in the future it will loop through all of them
    
    # Dynamically build the paths based on the selected patient
    edf_path = os.path.join(directorio_script, "..", "data", "mesa", "polysomnography", "edfs", f"mesa-sleep-{paciente_id}.edf")
    xml_path = os.path.join(directorio_script, "..", "data", "mesa", "polysomnography", "annotations-events-nsrr", f"mesa-sleep-{paciente_id}-nsrr.xml")
    
    edf_path = os.path.normpath(edf_path)
    xml_path = os.path.normpath(xml_path)

    # 2. Read the EDF header
    try:
        edf_info = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except FileNotFoundError:
        print(f"[ERROR] EDF file not found at: {edf_path}")
        return

    print("\n[INFO] Original Metadata:")
    print(f"  - Total channels: {len(edf_info.ch_names)}")
    print(f"  - Sampling rate: {edf_info.info['sfreq']} Hz")

    # Search for required ECG and EEG channels (EKG, EEG1, EEG2, and EEG3)
    ecg_canales = [c for c in edf_info.ch_names if ("ECG" in c.upper() or "EKG" in c.upper()) and "OFF" not in c.upper()]
    eeg_canales = [c for c in edf_info.ch_names if "EEG" in c.upper() and "OFF" not in c.upper()]

    print(f"  - ECG found: {ecg_canales}")
    print(f"  - EEG found: {eeg_canales}")

    # Process the data
    loader = DataLoader(target_fs=200.0)
    preprocessor = SignalPreprocessor(l_freq=0.5, h_freq=30.0)

    try:
        # Load and clean the signals
        raw_procesado = loader.load_and_standardize(
            edf_path=edf_path, 
            eeg_channels=eeg_canales, 
            ecg_channels=ecg_canales
        )
        
        # Attach the physician's notes from the XML file
        raw_procesado = loader.load_annotations(raw_procesado, xml_path)
        # Apply Band-pass (0.5-30Hz)
        raw_limpio = preprocessor.apply_filters(raw_procesado)

        # Extract matrix for NumPy
        matriz_datos = raw_limpio.get_data()
    
    except Exception as e:
        print(f"\n[ERROR] Failure during standardization: {e}")

if __name__ == "__main__":
    main()