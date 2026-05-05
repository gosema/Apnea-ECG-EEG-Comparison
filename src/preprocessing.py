import mne

class SignalPreprocessor:
    # Esta clase se encarga de aplicar los filtros digitales necesarios para limpiar las señales fisiológicas.
    # Recibe de parámetro los cortes de frecuencia para el filtro pasa-banda (l.freq y h_freq) del ruido y la frecuencia de la red eléctrica a eliminar.
    def __init__(self, l_freq=0.5, h_freq=30.0, notch_freq=50.0):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
    # Recibe el objeto nme (ondas cerebrales o cardíacas) y devuelve el mismo objeto pero con las señales filtradas.
    def apply_filters(self, raw):        
        # Implementación del filtro Notch para la interferencia de 60 Hz propia de la red electrica en EEUU
        raw.notch_filter(freqs=self.notch_freq, picks='all', verbose=False)
        
        # Implementación del filtro Pasa-banda mantiene solo la información relevante (0.5 - 30 Hz), quedando solo la actividad cerebral y cardiaca importante
        # Utilizamos un filtro FIR  que es el estándar en PDS y proyectos relacionados
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                   picks='all', method='fir', phase='zero', verbose=False)
        return raw