import mne

class SignalPreprocessor:
    # This class is responsible for applying the digital filters required to clean physiological signals.
    # It receives as parameters the frequency cutoffs for the band-pass filter (l_freq and h_freq) and the power line frequency to be removed.
    def __init__(self, l_freq=0.5, h_freq=30.0, notch_freq=50.0):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
    # It receives the MNE object (brain or cardiac waves) and returns the same object with the filtered signals.
    def apply_filters(self, raw):        
        # Implementation of a Notch filter for 60 Hz interference, typical of the US power grid.
        raw.notch_filter(freqs=self.notch_freq, picks='all', verbose=False)
        
        # Implementation of a Band-pass filter to retain only relevant information (0.5 - 30 Hz), preserving only significant brain and cardiac activity.
        # We use an FIR (Finite Impulse Response) filter, which is the standard in DSP (Digital Signal Processing) and related projects.
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                   picks='all', method='fir', phase='zero', verbose=False)
        return raw