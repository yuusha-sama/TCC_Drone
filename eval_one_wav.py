#!/usr/bin/env python3
from __future__ import annotations
import argparse, numpy as np, soundfile as sf
from acoustic_models.adapter import AcousticNNModel
from acoustic_models.features import ensure_mono_16k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    args = ap.parse_args()

    # carrega wav e classifica uma janela (inteira)
    x, sr = sf.read(args.wav, always_2d=False)
    x, sr = ensure_mono_16k(x, sr)

    nn = AcousticNNModel(
        model_path=args.model,
        labels_path="models/labels.txt",
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        sample_rate=16000,
        win_s=len(x)/sr,  # janela = arquivo inteiro
        hop_s=len(x)/sr,  # 1 chamada
        idx_drone=None,
        device_index=None,
        debug=False,
    )
    out = nn.classify_window(x, sr=sr)
    print("probs:", out["probs"])
    print("p(drone):", out["p_drone"])

if __name__ == "__main__":
    main()
