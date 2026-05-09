import mne
import scipy.signal as sgn

class SignalPreprocessor:
    # This class is responsible for applying the digital filters required to clean physiological signals.
    # It receives as parameters the frequency cutoffs for the band-pass filter (l_freq and h_freq)
    # The notch filter was not necessary, since applying the band-pass filter (up to 30 Hz) 
    # already eliminates the unwanted 50-60 Hz power line frequencies.
    def __init__(self, l_freq=0.5, h_freq=30.0):
        self.l_freq = l_freq
        self.h_freq = h_freq

    # It receives the MNE object (brain or cardiac waves) and returns the same object with the filtered signals.
    def apply_filters(self, raw):        
        # Convert from MNE to NumPy array to apply SciPy filters
        data = raw.get_data() 
        fs = raw.info['sfreq']
        # Nyquist frequency, which represents half of the sampling rate
        nyquist = 0.5 * fs
        low = self.l_freq / nyquist
        high = self.h_freq / nyquist
        
        #Butterworth filter and zero-phase filtering (filtfilt) needed to produce the filtered signal
        b, a = sgn.butter(4, [low, high], btype='band')
        data_filtered = sgn.filtfilt(b, a, data, axis=-1)
        # Reassign the filtered data back to the MNE object format
        raw._data = data_filtered

        return raw