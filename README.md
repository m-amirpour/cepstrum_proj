```markdown
# 🧠 Cepstrum-Based Speech Analysis 🎙️  
Extracting Pitch Contour and Formants from Speech Using Python and Signal Processing Techniques

---

## 🚀 Overview

This project implements **cepstral analysis** to extract two essential acoustic features from a recorded speech signal:

- 🎵 **Pitch Contour (Fundamental Frequency)**
- 🔍 **Formants (Vocal Tract Resonances: F1 and F2)**

Using **Python** and core **digital signal processing (DSP)** techniques, the system separates the **excitation source** (pitch) and the **vocal tract filter** (formants) based on the powerful **cepstrum representation**.

> ⚠️ Originally assigned using MATLAB — this project demonstrates that Python is equally capable for advanced academic DSP tasks.

---

## 📷 Demo Plots


  
  
  📈 Pitch contour of recorded speech (Hz vs. frame)



  
  
  🔍 Formants F1 & F2 extracted using LPC


---

## 🧪 How It Works

### ✅ Processing Pipeline:

1. **Record** or provide a short `.wav` speech file (`your_sentence.wav`)
2. **Frame** the signal into overlapping 30ms windows
3. **Apply Hamming window** to reduce spectral leakage
4. **Compute the cepstrum** via `IFFT(log(|FFT|))`
5. **Lifter** the cepstrum to:
   - Extract **pitch** from high-quefrency region
   - Extract **formants** using LPC from low-quefrency content
6. **Visualize** the features using `matplotlib`

---

## 📁 Project Structure

---

## 🛠️ Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `soundfile`

### 💡 Installation

pip install numpy scipy matplotlib soundfile

---

## 📚 Theoretical Background

Based on the classic speech production model:

**📌 Speech = Excitation × Vocal Tract**

In the frequency domain:

S(ω) = E(ω) × H(ω)
log|S(ω)| = log|E(ω)| + log|H(ω)|
Cepstrum = IFFT(log|S(ω)|)

### Key Ideas:

- **High-time** cepstral coefficients → **pitch**
- **Low-time** cepstral coefficients → **vocal tract filter** (formants)

### 📉 LPC for Formant Extraction

- **LPC** (Linear Predictive Coding) models speech as an all-pole filter  
- **Roots** of the LPC polynomial give **formant frequencies** (F1, F2)  
- **Angles** of complex roots → mapped to Hz using sampling rate

---

## 📈 Results

- ✔️ Accurate **pitch contour** extracted from raw audio  
- ✔️ **First two formants (F1, F2)** estimated per frame  
- ✔️ Output plots match **acoustic expectations** based on vowel characteristics

---

## 🧑‍🏫 Educational Context

This repository was developed for an academic **Natural Language Processing (NLP)** course project (Spring 1403–04 / 2025). It aligns with lessons on:

- Cepstrum theory  
- Speech signal modeling  
- Feature extraction for phoneme analysis  

> 📝 Originally implemented in MATLAB. This Python version improves accessibility and reproducibility.

---

## 🙌 Acknowledgments

- 📘 Lawrence Rabiner and Ronald Schafer – for foundational DSP theories  
- 🏫 K. N. Toosi University of Technology – NLP course team  
- 🐍 Python open-source community – for available DSP libraries

---

## 📬 Contact

Made with ❤️ by **Muhammad Mahdi Amirpour**  
🔗 [https://github.com/m-amirpour](https://github.com/m-amirpour)  
✉️ muhammadmahdiamirpour@gmail.com

---
```

