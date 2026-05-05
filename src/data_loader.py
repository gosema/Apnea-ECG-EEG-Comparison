import mne
import numpy as np
import warnings
import xml.etree.ElementTree as ET

# Suprimir warnings molestos de MNE al cargar EDFs
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DataLoader:
    def __init__(self, target_fs=100.0):
        """
        Inicializa el cargador de datos.
        
        :param target_fs: Frecuencia de muestreo objetivo (en Hz) para homogeneizar 
                          todas las señales. 100 Hz o 128 Hz suele ser suficiente 
                          para la mayoría de características espectrales del sueño.
        """
        self.target_fs = target_fs

    def load_and_standardize(self, edf_path, eeg_channels, ecg_channels):
        """
        Carga un archivo EDF y aplica la estandarización completa.
        """
        print(f"Procesando: {edf_path}...")
        
        # 1. Cargar el registro crudo (preload=True es necesario para modificar los datos)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Filtrar solo los canales que nos interesan (EEG y ECG)
        channels_to_keep = eeg_channels + ecg_channels
        
        # Verificar qué canales existen realmente en el EDF para evitar errores
        existing_channels = [ch for ch in channels_to_keep if ch in raw.ch_names]
        raw.pick(existing_channels)

        # 2. Armonizar la frecuencia de muestreo (Resampling)
        if raw.info['sfreq'] != self.target_fs:
            print(f"  - Remuestreando de {raw.info['sfreq']} Hz a {self.target_fs} Hz")
            raw.resample(self.target_fs)

        # 3. Establecer referencia común para el EEG
        # Usamos CAR (Common Average Reference) que calcula la media de todos los 
        # electrodos EEG y la resta, mitigando las diferencias físicas de los montajes.
        eeg_existentes = [ch for ch in eeg_channels if ch in raw.ch_names]
        if eeg_existentes:
            # Le decimos a MNE cuáles son los canales EEG
            raw.set_channel_types({ch: 'eeg' for ch in eeg_existentes})
            print("  - Aplicando referencia común promedio (CAR) al EEG")
            raw.set_eeg_reference('average', projection=False, verbose=False)

