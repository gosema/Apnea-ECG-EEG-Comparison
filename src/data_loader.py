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

        # 4. Normalizar Amplitudes (Z-score normalization)
        # Esto asegura que pacientes con señales inherentemente más fuertes/débiles 
        # se evalúen en la misma escala (media 0, desviación estándar 1).
        print("  - Normalizando amplitudes (Z-score)")
        data = raw.get_data() # Devuelve un array de numpy (canales x muestras)
        
        # Normalizamos canal por canal
        for i in range(data.shape[0]):
            media = np.mean(data[i, :])
            std = np.std(data[i, :])
            if std > 0: # Prevenir división por cero si un canal está "plano"
                data[i, :] = (data[i, :] - media) / std
                
        # Reasignamos los datos normalizados al objeto raw
        raw._data = data

        print("  - ¡Procesamiento inicial completado con éxito!")
        return raw

    def load_annotations(self, raw, xml_path):
        """
        Lee un archivo XML (formato NSRR o Profusion) y superpone las etiquetas
        sobre el objeto de señales en crudo (raw) de MNE.
        """
        print(f"  - Cargando anotaciones desde: {xml_path}")
        
        try:
            # Parsear el archivo XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"    [ERROR] No se pudo leer el archivo XML: {e}")
            return raw
            
        onsets = []
        durations = []
        descriptions = []
        
        # Buscar todos los eventos etiquetados en el archivo
        # En el formato NSRR, la etiqueta suele ser <ScoredEvent>
        for event in root.iter('ScoredEvent'):
            # Diferentes bases de datos usan 'Name' o 'EventConcept' para el nombre
            name_node = event.find('Name')
            if name_node is None:
                name_node = event.find('EventConcept')
                
            start_node = event.find('Start')
            duration_node = event.find('Duration')
            
            # Si el evento tiene nombre, inicio y duración, lo extraemos
            if name_node is not None and start_node is not None and duration_node is not None:
                desc = name_node.text
                try:
                    start = float(start_node.text)
                    duration = float(duration_node.text)
                    
                    onsets.append(start)
                    durations.append(duration)
                    descriptions.append(desc)
                except ValueError:
                    # Ignorar si hay algún texto corrupto que no sea un número
                    continue
                    
        if len(onsets) > 0:
            # Crear el objeto de anotaciones de MNE
            # Se usa el orig_time del archivo EDF para que las horas coincidan exactamente
            annotations = mne.Annotations(
                onset=onsets, 
                duration=durations, 
                description=descriptions,
                orig_time=raw.info['meas_date']
            )
            
            # Acoplar las etiquetas a la señal fisiológica
            raw.set_annotations(annotations)
            print(f"  - [ÉXITO] Se han acoplado {len(onsets)} etiquetas clínicas a la señal.")
        else:
            print("  - [ADVERTENCIA] No se encontraron eventos legibles en el XML.")
            
        return raw