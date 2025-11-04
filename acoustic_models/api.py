from typing import Dict
import numpy as np
from .features import logmel_spectrogram, ensure_mono_16k
from .keras_backend import KerasBackend

def load_backend(kind: str, model_path: str):
    """Keras-only: ignora 'kind' e sempre retorna KerasBackend."""
    return KerasBackend(model_path)

def classify_window(pcm: np.ndarray, sr: int, backend, *,
                    n_fft=1024, n_mels=64, hop_in_fft=None,
                    idx_drone=None) -> Dict[str, float]:
    x, sr = ensure_mono_16k(pcm, sr)
    hop = (n_fft//4) if hop_in_fft is None else hop_in_fft
    LM = logmel_spectrogram(x, sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    LM = (LM - LM.mean())/(LM.std() + 1e-6)
    X = LM[None, ..., None].astype(np.float32)  # [1, T, F, 1]
    y = backend.predict_proba(X)                # [1, C]
    y = np.squeeze(y).astype(float)             # [C] ou escalar
    if np.ndim(y) == 0:
        p_drone = float(y); probs = [p_drone]
    else:
        C = int(len(y))
        if C == 1:
            p_drone = float(y[0]); probs = [p_drone]
        else:
            if idx_drone is None: idx_drone = 1
            p_drone = float(y[int(idx_drone)]); probs = list(map(float, y.tolist()))
    return {"p_drone": p_drone, "probs": probs}
