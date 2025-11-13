from typing import Dict, Tuple
import numpy as np
from .features import logmel_spectrogram, ensure_mono_16k
from .keras_backend import KerasBackend

def _pad_or_crop_time(LM: np.ndarray, t_frames: int) -> np.ndarray:
    """LM: [T,F] -> ajusta T para t_frames."""
    if t_frames is None:
        return LM
    T, F = LM.shape
    if T == t_frames:
        return LM
    if T > t_frames:
        s = (T - t_frames) // 2
        return LM[s:s+t_frames, :]
    out = np.zeros((t_frames, F), dtype=LM.dtype)
    out[:min(T, t_frames)] = LM[:min(T, t_frames)]
    return out

def load_backend(kind: str, model_path: str):
    """Keras-only: ignora 'kind' e sempre retorna KerasBackend."""
    return KerasBackend(model_path)

def classify_window(pcm: np.ndarray, sr: int, backend, *,
                    n_fft=1024, n_mels=64, hop_in_fft=None,
                    idx_drone=None) -> Dict[str, float]:
    # 1) mono/16k
    x, sr = ensure_mono_16k(pcm, sr)

    # 2) espectrograma log-mel
    hop = (n_fft//4) if hop_in_fft is None else hop_in_fft

    # Ajuste de F (n_mels) se o modelo especificar
    T_exp, F_exp = (backend.expected_tf() if hasattr(backend, "expected_tf") else (None, None))
    if F_exp is not None:
        n_mels = F_exp

    LM = logmel_spectrogram(x, sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    LM = (LM - LM.mean())/(LM.std() + 1e-6)

    # 3) pad/crop para T esperado
    LM = _pad_or_crop_time(LM, T_exp)

    # 4) batch e canal
    X = LM[None, ..., None].astype(np.float32)  # [1, T, F, 1]

    # 5) inferência
    y = backend.predict_proba(X)                # [1, C] ou [1] ou escalar
    y = np.squeeze(y).astype(float)

    # 6) extrair p_drone
    if np.ndim(y) == 0:
        p_drone = float(y); probs = [p_drone]
    else:
        C = int(len(y))
        if C == 1:
            p_drone = float(y[0]); probs = [p_drone]
        else:
            if idx_drone is None: idx_drone = 1
            p_drone = float(y[int(idx_drone)])
            probs = list(map(float, y.tolist()))
    return {"p_drone": p_drone, "probs": probs}
