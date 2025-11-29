# DB/audio_mfcc_utils.py
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa


def extract_mfcc_from_file(
    path: Path,
    target_sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extrai MFCC de um arquivo .wav e retorna um vetor de features fixo
    (média + desvio padrão de cada coeficiente).
    """
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    features = np.concatenate([mfcc_mean, mfcc_std])

    return features.astype(np.float32)


def extract_mfcc_from_buffer(
    audio_buffer: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Extrai MFCC de um pedaço de áudio já em memória (para uso em tempo real).
    """
    # Se veio com mais de um canal, vira mono
    if audio_buffer.ndim > 1:
        audio_buffer = audio_buffer.mean(axis=1)

    mfcc = librosa.feature.mfcc(
        y=audio_buffer,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])

    return features.astype(np.float32)
