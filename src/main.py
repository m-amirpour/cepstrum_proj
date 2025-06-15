"""
Pitch and Formant Contour Extraction and Visualization using Cepstrum and LPC

This script reads a WAV audio file, computes the pitch contour using the cepstrum method,
estimates formant frequencies using Linear Predictive Coding (LPC), and plots both pitch
and the first three formants over time.

Author: Your Name
Date: 2025-06-15
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hamming
from scipy.linalg import toeplitz

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to the input signal to amplify high frequencies.

    Parameters:
        signal (np.ndarray): Input audio signal.
        coeff (float): Pre-emphasis coefficient (default 0.97).

    Returns:
        np.ndarray: Pre-emphasized signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def compute_cepstrum_pitch(signal: np.ndarray, fs: int,
                           frame_size: float = 0.04, frame_shift: float = 0.01,
                           fmin: int = 70, fmax: int = 270) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pitch contour of the signal using the cepstrum method.

    Parameters:
        signal (np.ndarray): Mono audio signal normalized between -1 and 1.
        fs (int): Sampling frequency in Hz.
        frame_size (float): Frame length in seconds (default 40 ms).
        frame_shift (float): Frame shift in seconds (default 10 ms).
        fmin (int): Minimum expected pitch frequency in Hz (default 70 Hz).
        fmax (int): Maximum expected pitch frequency in Hz (default 270 Hz).

    Returns:
        times (np.ndarray): Time stamps (seconds) for each frame center.
        pitches (np.ndarray): Estimated pitch frequencies (Hz), 0 if unvoiced.
    """
    win_length = int(frame_size * fs)
    hop_length = int(frame_shift * fs)
    nfft = 2**int(np.ceil(np.log2(win_length)))  # FFT length (next power of two)
    window = hamming(win_length)

    pitches = []
    times = []

    for start in range(0, len(signal) - win_length, hop_length):
        frame = signal[start:start + win_length] * window

        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(frame, nfft))**2
        spectrum[spectrum == 0] = np.finfo(float).eps  # Avoid log(0)

        # Log spectrum and real cepstrum
        log_spectrum = np.log(spectrum)
        cepstrum = np.fft.ifft(log_spectrum).real

        # Define quefrency range corresponding to plausible pitch periods
        quefrency = np.arange(len(cepstrum)) / fs
        min_quefrency = 1 / fmax
        max_quefrency = 1 / fmin

        valid_range = np.where((quefrency >= min_quefrency) & (quefrency <= max_quefrency))[0]

        # Find peak in valid quefrency range
        peak_index = valid_range[np.argmax(cepstrum[valid_range])]
        pitch_period = quefrency[peak_index]
        pitch_freq = 1 / pitch_period

        # Simple voiced/unvoiced decision threshold on cepstrum peak amplitude
        if cepstrum[peak_index] < 0.1:
            pitch_freq = 0

        pitches.append(pitch_freq)
        times.append((start + win_length / 2) / fs)  # Center time of frame

    return np.array(times), np.array(pitches)

def lpc_formants(frame: np.ndarray, fs: int, order: int = 12) -> list[float]:
    """
    Estimate formant frequencies from a single audio frame using LPC.

    Parameters:
        frame (np.ndarray): Windowed audio frame.
        fs (int): Sampling frequency in Hz.
        order (int): LPC order (default 12).

    Returns:
        list[float]: Sorted list of formant frequencies in Hz.
    """
    # Apply pre-emphasis to emphasize higher frequencies
    emphasized = pre_emphasis(frame)

    # Compute autocorrelation of the frame
    autocorr = np.correlate(emphasized, emphasized, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags

    # Construct Toeplitz matrix from autocorrelation
    R = autocorr[:order + 1]
    r = R[1:]
    R_matrix = toeplitz(R[:-1])

    try:
        # Solve LPC coefficients: R_matrix * a = r
        a = np.linalg.solve(R_matrix, r)
    except np.linalg.LinAlgError:
        # Return empty list if matrix is singular
        return []

    a = np.concatenate(([1], -a))  # LPC polynomial coefficients

    # Find roots of LPC polynomial
    roots = np.roots(a)

    # Keep roots with positive imaginary part (complex conjugates)
    roots = [r for r in roots if np.imag(r) >= 0.01]

    # Convert roots to angles (radians), then to frequencies (Hz)
    angz = np.angle(roots)
    formants = sorted(angz * (fs / (2 * np.pi)))

    # Filter formants to reasonable frequency range (90 Hz to Nyquist)
    formants = [f for f in formants if 90 < f < fs / 2]

    return formants

def compute_formants_contour(signal: np.ndarray, fs: int,
                             frame_size: float = 0.04, frame_shift: float = 0.01,
                             lpc_order: int = 12) -> tuple[np.ndarray, list[list[float]]]:
    """
    Compute formant frequencies over time for the entire signal.

    Parameters:
        signal (np.ndarray): Mono audio signal normalized between -1 and 1.
        fs (int): Sampling frequency in Hz.
        frame_size (float): Frame length in seconds (default 40 ms).
        frame_shift (float): Frame shift in seconds (default 10 ms).
        lpc_order (int): LPC order (default 12).

    Returns:
        times (np.ndarray): Time stamps (seconds) for each frame center.
        formants_list (list of lists): List of formants per frame.
    """
    win_length = int(frame_size * fs)
    hop_length = int(frame_shift * fs)
    window = hamming(win_length)

    formants_list = []
    times = []

    for start in range(0, len(signal) - win_length, hop_length):
        frame = signal[start:start + win_length] * window
        formants = lpc_formants(frame, fs, order=lpc_order)
        formants_list.append(formants)
        times.append((start + win_length / 2) / fs)

    return np.array(times), formants_list

def main():
    # Load WAV file
    sample_rate, audio_data = wavfile.read('data/snlp-project-input.wav')

    # Convert stereo to mono by averaging channels if needed
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    # Normalize audio to float32 in range [-1, 1]
    if audio_data.dtype == 'int16':
        audio_data = audio_data / 32768.0
    elif audio_data.dtype == 'int32':
        audio_data = audio_data / 2147483648.0

    # Compute pitch contour using cepstrum
    times_pitch, pitches = compute_cepstrum_pitch(audio_data, sample_rate)

    # Compute formant contours using LPC
    times_formants, formants_contour = compute_formants_contour(audio_data, sample_rate)

    # Extract first three formants per frame for plotting; fill missing with NaN
    f1 = [f[0] if len(f) > 0 else np.nan for f in formants_contour]
    f2 = [f[1] if len(f) > 1 else np.nan for f in formants_contour]
    f3 = [f[2] if len(f) > 2 else np.nan for f in formants_contour]

    # Plot pitch and formants
    plt.figure(figsize=(12, 6))

    plt.plot(times_pitch, pitches, label='Pitch (Hz)', color='blue')
    plt.plot(times_formants, f1, label='Formant 1 (Hz)', color='red', linestyle='--')
    plt.plot(times_formants, f2, label='Formant 2 (Hz)', color='green', linestyle='--')
    plt.plot(times_formants, f3, label='Formant 3 (Hz)', color='orange', linestyle='--')

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch and Formant Contours')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()

