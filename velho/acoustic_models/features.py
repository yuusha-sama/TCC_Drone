import numpy as np
from scipy.signal import get_window, resample_poly

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10**(m/2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels=40, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    freqs = np.linspace(0, sr/2.0, n_fft//2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(1, len(hz)-1):
        f_l, f_c, f_r = hz[i-1], hz[i], hz[i+1]
        left = np.logical_and(freqs >= f_l, freqs <= f_c)
        fb[i-1, left] = (freqs[left] - f_l) / (f_c - f_l + 1e-10)
        right = np.logical_and(freqs >= f_c, freqs <= f_r)
        fb[i-1, right] = (f_r - freqs[right]) / (f_r - f_c + 1e-10)
    enorm = 2.0 / (hz[2:n_mels+2] - hz[:n_mels])
    fb *= enorm[:, None]
    return fb

def stft(y, n_fft=1024, hop_length=256, win="hann"):
    w = get_window(win, n_fft, fftbins=True).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop_length if len(y) >= n_fft else 1
    if n_frames < 1:
        n_frames = 1
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        seg = np.zeros(n_fft, dtype=np.float32)
        if start < len(y):
            seg[:max(0, min(n_fft, len(y)-start))] = y[start:end]
        frames[i] = seg * w
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec

def logmel_spectrogram(y, sr, n_fft=1024, hop_length=256, n_mels=64, fmin=20.0, fmax=None, eps=1e-10):
    y = np.asarray(y, dtype=np.float32).flatten()
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**2
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    M = np.dot(S, fb.T)
    M = np.maximum(M, eps)
    return np.log(M)

def ensure_mono_16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != 16000:
        from fractions import Fraction
        frac = Fraction(16000, int(sr)).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        x = resample_poly(x, up, down).astype(np.float32)
        sr = 16000
    return x, sr
