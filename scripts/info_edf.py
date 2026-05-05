import mne
import sys
import os

# 1. Configurar el path para que Python encuentre la carpeta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importamos nuestra clase
from src.data_loader import DataLoader

def main():
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # --- CONFIGURACIÓN DEL PACIENTE A PRUEBA ---
    paciente_id = "0006"  # Cambia esto a "0033" o "0010" para probar otros
    
    # Construir las rutas dinámicamente según el paciente elegido
    edf_path = os.path.join(directorio_script, "..", "data", "mesa", "polysomnography", "edfs", f"mesa-sleep-{paciente_id}.edf")
    xml_path = os.path.join(directorio_script, "..", "data", "mesa", "polysomnography", "annotations-events-nsrr", f"mesa-sleep-{paciente_id}-nsrr.xml")
    
    edf_path = os.path.normpath(edf_path)
    xml_path = os.path.normpath(xml_path)
    # ------------------------------------------

    print(f"--- Explorando paciente: {paciente_id} ---")

    # 2. Leer SOLO la cabecera del EDF (preload=False)
    try:
        edf_info = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except FileNotFoundError:
        print(f"[ERROR] No se encuentra el archivo EDF en: {edf_path}")
        return

    print("\n[INFO] Metadatos Originales:")
    print(f"  - Canales totales: {len(edf_info.ch_names)}")
    print(f"  - Frecuencia de muestreo: {edf_info.info['sfreq']} Hz")

    # 3. Buscar canales (Ignorando los que contengan "OFF" o "STATUS")
    ecg_canales = [c for c in edf_info.ch_names if ("ECG" in c.upper() or "EKG" in c.upper()) and "OFF" not in c.upper()]
    eeg_canales = [c for c in edf_info.ch_names if "EEG" in c.upper() and "OFF" not in c.upper()]

    print(f"\n[INFO] Búsqueda de señales fisiológicas puras:")
    print(f"  - ECG encontrados: {ecg_canales}")
    print(f"  - EEG encontrados: {eeg_canales}")

    if not ecg_canales and not eeg_canales:
        print("\n[ADVERTENCIA] No se encontraron canales relevantes. Abortando.")
        return

   