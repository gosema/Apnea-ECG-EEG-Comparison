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
