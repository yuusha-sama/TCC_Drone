#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_one_wav_sliding_window.py — Avaliação avançada usando janelas deslizantes.
Compatível com modelos binários (sigmoid).
Mostra p(drone) por janela mas dá uma decisão final clara:
    DRONE  ou  NÃO É DRONE
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path

from acoustic_models.keras_backend import KerasBackend
from acoustic_models.features import logmel_spectrogram, ensure_mono_16k

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}


# ----------------------------------------------------------
# Localizar arquivos
# ----------------------------------------------------------

def list_audio_files(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo/pasta não encontrado: {path}")

    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]

    if p.is_dir():
        files = []
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in AUDIO_EXTS:
                files.append(fp)
        if not files:
            raise RuntimeError(f"Nenhum áudio válido encontrado em: {path}")
        return files

    raise RuntimeError(f"Caminho inválido: {path}")


# ----------------------------------------------------------
# Avaliação por janelas deslizantes
# ----------------------------------------------------------

def eval_sliding_windows(model: KerasBackend, wav_path, threshold=0.6):
    print("\n========================================")
    print("🎧 Arquivo:", wav_path)
    print("========================================")

    try:
        audio, sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {wav_path}: {e}")
        return None

    # Normaliza para mono 16kHz
    audio, sr = ensure_mono_16k(audio, sr)

    # Log-mel
    LM = logmel_spectrogram(audio, sr, n_fft=1024, hop_length=256, n_mels=64)
    LM = (LM - LM.mean()) / (LM.std() + 1e-6)

    T_exp, F_exp = model.expected_tf()

    # Ajuste bandas (F)
    if F_exp is not None:
        LM = LM[:, :F_exp]

    T_total, F = LM.shape

    if T_exp is None:
        raise RuntimeError("O modelo não retornou T_exp.")

    step = max(1, T_exp // 4)  # janelas com sobreposição
    probs = []

    # Percorrer log-mel inteiro
    for start in range(0, max(1, T_total - T_exp + 1), step):
        sub = LM[start:start+T_exp, :]

        if sub.shape[0] < T_exp:
            pad = np.zeros((T_exp, F), dtype=sub.dtype)
            pad[:sub.shape[0]] = sub
            sub = pad

        X = sub[None, :, :, None]

        # Inferência: binário → y = p(drone)
        y = model.predict_proba(X)
        y = np.squeeze(y).astype(float)
        p_drone = float(y)
        probs.append(p_drone)

    # Mostrar janelas
    print("\n🔎 p(drone) por janela:")
    for i, p in enumerate(probs):
        print(f"  janela {i:02d}: {p:.4f}")

    # Cálculos finais
    p_max = max(probs)
    p_med = sum(probs) / len(probs)

    # --------------------------------------------------
    # DECISÃO FINAL
    # --------------------------------------------------
    is_drone = p_max >= threshold

    print("\n📌 RESULTADO FINAL")
    print("------------------------")
    print(f"p(drone)_MAX = {p_max:.4f}")
    print(f"p(drone)_MED = {p_med:.4f}")

    if is_drone:
        print(f"\n🎯 DECISÃO: **DRONE DETECTADO** (p={p_max:.4f})\n")
    else:
        print(f"\n🟦 DECISÃO: **NÃO É DRONE** (p={p_max:.4f})\n")

    return is_drone


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Avaliação com janelas deslizantes")
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    print("🔄 Carregando modelo Keras...")
    model = KerasBackend(args.model)

    print("📁 Lendo arquivos...")
    files = list_audio_files(args.wav)

    for fp in files:
        eval_sliding_windows(model, fp, threshold=args.threshold)


if __name__ == "__main__":
    main()
