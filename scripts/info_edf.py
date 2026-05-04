import mne

edf_path = "f:/ProyectoPDS/data/data/mesa/polysomnography/edfs/mesa-sleep-0006.edf"

#Leer el edf
edf_prueba = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
print(edf_prueba.ch_names)
print(edf_prueba.info["sfreq"])

# Buscar canales por nombre
ecg_canal = [c for c in edf_prueba.ch_names if "ECG" in c.upper() or "EKG" in c.upper()]
eeg_canal = [c for c in edf_prueba.ch_names if "EEG" in c.upper()]

print("ECG:", ecg_canal)
print("EEG:", eeg_canal)

# Extraer señal de un canal
if ecg_canal:
    ecg_senial = edf_prueba.get_data(picks=[ecg_canal[0]])[0]
if eeg_canal:
    eeg_senial = edf_prueba.get_data(picks=[eeg_canal[0]])[0]