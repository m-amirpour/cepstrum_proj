import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import get_window
from scipy.fftpack import fft, ifft
from numpy.linalg import solve

# Load recorded sentence
signal, sr = sf.read("data/snlp-project-input.wav")
if len(signal.shape) > 1:
    signal = signal[:, 0]  # if stereo, keep one channel

# Frame parameters
frame_length = int(0.03 * sr)  # 30ms
hop_length = int(0.015 * sr)   # 15ms
window = get_window("hamming", frame_length)

# Parameters for pitch estimation
min_pitch = 60  # Hz
max_pitch = 400  # Hz
min_quef = int(sr / max_pitch)
max_quef = int(sr / min_pitch)

pitch_values = []
formant_f1 = []
formant_f2 = []

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2:]

def lpc_coefficients(frame, order):
    R = autocorr(frame)
    if R[0] == 0:
        return np.ones(order + 1)
    R_matrix = np.array([[R[abs(i-j)] for j in range(order)] for i in range(order)])
    r_vector = -R[1:order+1]
    epsilon = 1e-6
    R_matrix += epsilon * np.eye(order)
    a = solve(R_matrix, r_vector)
    return np.concatenate(([1], a))

def extract_formants(a, sr):
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) > 0.01]
    angles = np.angle(roots)
    formants = sorted(angles * (sr / (2 * np.pi)))
    return formants[:2]

# Frame-by-frame analysis
for i in range(0, len(signal) - frame_length, hop_length):
    frame = signal[i:i + frame_length]
    frame = frame * window

    # ---------------- PITCH (via high-time liftering in cepstrum)
    spectrum = fft(frame)
    log_magnitude = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.real(ifft(log_magnitude))

    high_liftered = np.copy(cepstrum)
    high_liftered[:min_quef] = 0
    high_liftered[max_quef:] = 0

    peak_index = np.argmax(high_liftered)
    pitch = sr / peak_index if peak_index > 0 else 0
    pitch_values.append(pitch)

    # ---------------- FORMANTS (via LPC)
    try:
        a = lpc_coefficients(frame, order=12)
        f1, f2 = extract_formants(a, sr)
        formant_f1.append(f1)
        formant_f2.append(f2)
    except:
        formant_f1.append(0)
        formant_f2.append(0)

# Plot pitch contour
plt.figure(figsize=(10, 4))
plt.plot(pitch_values, label="Pitch (Hz)")
plt.title("Pitch Contour via Cepstrum")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pitch_contour.png")
plt.show()

# Plot formants
plt.figure(figsize=(10, 4))
plt.plot(formant_f1, label='Formant F1')
plt.plot(formant_f2, label='Formant F2')
plt.title("Formants via LPC")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("formants.png")
plt.show()

