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
