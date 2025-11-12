#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_one_wav.py — Avalia um único arquivo WAV (ou todos os WAVs dentro de uma pasta)
usando o modelo acústico Keras do seu TCC.

Uso:
  python eval_one_wav.py --model models/audio_classifier.keras --wav caminho/do/audio.wav
  python eval_one_wav.py --model models/audio_classifier.keras --wav pasta/com/audios
"""

import argparse
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Importando do seu projeto:
from acoustic_models.api import classify_window
from acoustic_models.keras_backend import KerasBackend

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}


def list_audio_files(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo/pasta não encontrado: {path}")

    # Caso seja arquivo único
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]

    # Caso seja pasta
    if p.is_dir():
        files = []
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in AUDIO_EXTS:
                files.append(fp)
        if not files:
            raise RuntimeError(f"Nenhum áudio válido encontrado em: {path}")
        return files

    raise RuntimeError(f"Caminho inválido: {path}")


def eval_file(model, wav_path, idx_drone=None):
    """Avalia um único arquivo WAV e retorna p_drone."""
    try:
        audio, sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {wav_path}: {e}")
        return None

    res = classify_window(audio, sr, backend=model, idx_drone=idx_drone)
    p = float(res.get("p_drone", 0.0))

    print(f"\n📄 Arquivo: {wav_path}")
    print(f"🎧 Probabilidade drone = {p:.4f}")
    print(f"🔢 Todas as probabilidades = {res.get('probs')}")

    return p


def main():
    ap = argparse.ArgumentParser(description="Avaliação de um ou vários áudios WAV.")
    ap.add_argument("--model", required=True, help="Caminho do modelo .keras/.h5")
    ap.add_argument("--wav", required=True, help="Caminho do arquivo ou pasta de áudios")
    ap.add_argument("--idx_drone", type=int, default=None, help="Índice da classe drone (opcional)")

    args = ap.parse_args()

    print("🔄 Carregando modelo Keras...")
    model = KerasBackend(args.model)

    print("📁 Coletando arquivos...")
    files = list_audio_files(args.wav)

    print(f"🎵 Encontrados {len(files)} arquivo(s) de áudio.\n")

    results = []

    for fp in files:
        p = eval_file(model, fp, idx_drone=args.idx_drone)
        if p is not None:
            results.append((fp, p))

    if len(results) > 1:
        print("\n============== RESUMO ==============")
        for fp, p in results:
            print(f"{fp.name:40s}  →  p(drone)={p:.4f}")


if __name__ == "__main__":
    main()
