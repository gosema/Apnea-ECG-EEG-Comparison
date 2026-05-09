import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def process_signal_fourier(signal, fs, type="EEG"):
    if type == "ECG":
        # Detect R-peaks (index of each heartbeat)
        # threshold -min vertical distance between neighbouring samples- is set to half of the standard deviation of the signal to avoid false positives
        # distance -min horizontal distance (in samples) between neighbouring peaks- is set to 0.6 seconds (fs*0.6) to avoid detecting multiple peaks within the same heartbeat
        r_peaks, _ = find_peaks(signal, threshold=np.std(signal) * 0.5, distance=fs*0.6)  # Assuming a minimum heart rate of 60 bpm
        return process_ecg(r_peaks, fs)
    
    return process_eeg(signal, fs)

def process_eeg(signal, fs):
    N = len(signal)
    # Aplly a Hanning window to minimize spectral leakage (Gibbs phenomenon)
    windowed = signal * np.hanning(N)
    
    # rfft returns only the non-redundant positive frequencies
    # Result length: (N/2) + 1
    yf = rfft(windowed)
    xf = rfftfreq(N, 1/fs)
    
    # Calculate Magnitude (or Power Spectral Density)
    # Multiply by 2 (except for DC and Nyquist) to conserve energy
    # because we dropped the negative frequencies.
    psd = (np.abs(yf)**2) / (fs * N)
    psd[1:-1] *= 2 
    
    return xf, psd

def process_ecg(r_peaks, fs):
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    
    # Interpolate to get a uniformly sampled signal
    t = np.cumsum(rr_intervals)
    interp_func = interp1d(t, rr_intervals, kind='cubic')
    
    # Resample at 4 Hz (typical for HRV analysis)
    resampled_t = np.arange(t[0], t[-1], 1/fs)
    resampled_rr = interp_func(resampled_t)

    # Apply FFT to the resampled RR intervals to analyze HRV in the frequency domain
    N = len(resampled_rr)
    # Remove DC component (mean) to focus on variability
    resampled_rr -= np.mean(resampled_rr)

    yf = rfft(resampled_rr * np.hanning(N))  # Apply Hanning window
    xf = rfftfreq(N, 1/fs)

    # Power Spectral Density of HRV
    psd = (np.abs(yf)**2) / (fs * N)
    psd[1:-1] *= 2  # Conservation of energy for non-DC and non-Nyquist frequencies

    # Feature Extraction for Apnea
    # Bands: LF (0.04-0.15 Hz); HF (0.15-0.4 Hz); VLF (0.003-0.04 Hz)
    lf_band = (xf >= 0.04) & (xf < 0.15)
    hf_band = (xf >= 0.15) & (xf < 0.4)
    vlf_band = (xf >= 0.003) & (xf < 0.04)
    lf_power = np.sum(psd[lf_band])
    hf_power = np.sum(psd[hf_band])
    vlf_power = np.sum(psd[vlf_band])

    return {
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_power / hf_power if hf_power > 0 else np.inf,
        "total_power": lf_power + hf_power + vlf_power
    }