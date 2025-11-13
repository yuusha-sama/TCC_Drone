#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
live_test.py — Teste acústico ao vivo com janela deslizante e votação.

Funcionalidades:
- --list-devices: lista dispositivos de áudio e sai (sem precisar de --model)
- --debug: mostra informações internas (rms, gain, probs) vindas do adapter
- Decisão de DRONE baseada em múltiplas janelas acima de um limiar (threshold)

Lógica de decisão:
- Mantém um histórico das últimas N janelas (history, deque).
- Em cada nova janela, conta quantas têm p(drone) >= threshold.
- Se pelo menos `min_windows_above` janelas estiverem acima do threshold,
  considera DRONE DETECTADO.

Parâmetros recomendados para uso real:
- threshold ≈ 0.25 para modo ao vivo (mais sensível)
- min_windows_above = 2 (duas janelas consecutivas/sobrepostas)
"""

from __future__ import annotations
import argparse
import time
from collections import deque

import sounddevice as sd

from .adapter import AcousticNNModel


def list_devices():
    """Lista os dispositivos de áudio disponíveis."""
    print("=== Dispositivos de áudio disponíveis ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(
            f"{i:2d} | {d['name']} "
            f"(in={d['max_input_channels']}, out={d['max_output_channels']}, "
            f"default_sr={d.get('default_samplerate', 'N/A')})"
        )


def main():
    ap = argparse.ArgumentParser(description="Teste acústico ao vivo com janela deslizante")

    # model NÃO é required aqui; a gente valida depois, só se não for --list-devices
    ap.add_argument("--model", help="Caminho do modelo Keras (.keras/.h5)")
    ap.add_argument("--labels", default="models/labels.txt")

    # Taxa de amostragem do dispositivo de captura
    ap.add_argument("--rate", type=int, default=16000)

    # Tamanho da janela (segundos) e hop (segundos)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)

    # Parâmetros de STFT/Mel
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)

    # Índice da classe "drone" (se None, assume saída binária [p_not_drone, p_drone]
    ap.add_argument("--idx_drone", type=int, default=None)

    # Limiar de decisão (calibrado para o modo ao vivo)
    ap.add_argument("--threshold", type=float, default=0.25)

    # Índice do dispositivo de áudio (ver com --list-devices)
    ap.add_argument("--device", type=int, default=None)

    # Flags especiais
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Mostra informações internas (rms, gain, probs) do modelo"
    )

    args = ap.parse_args()

    # Caso especial: só listar devices e sair
    if args.list_devices:
        list_devices()
        return

    # A partir daqui, precisamos de modelo
    if not args.model:
        ap.error("--model é obrigatório (exceto quando usado com --list-devices).")

    print("🔄 Inicializando detector acústico ao vivo...\n")

    nn = AcousticNNModel(
        model_path=args.model,
        labels_path=args.labels,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        sample_rate=args.rate,
        win_s=args.win,
        hop_s=args.hop,
        idx_drone=args.idx_drone,
        device_index=args.device,
        threshold=args.threshold,
        debug=args.debug,  # passa debug pro adapter
    )

    # Histórico das últimas janelas (~4s se hop=0.5 e maxlen=8)
    history = deque(maxlen=8)

    # número mínimo de janelas acima do threshold para considerar DRONE
    # (ajusta se quiser mais ou menos sensível)
    min_windows_above = 2

    def on_res(r: dict):
        """
        Callback chamado pelo adapter a cada nova janela classificada.
        r deve conter:
            - "p_drone": probabilidade de drone (float)
        """
        p = float(r.get("p_drone", 0.0))
        history.append(p)

        if history:
            p_max = max(history)
            p_med = sum(history) / len(history)
        else:
            p_max = 0.0
            p_med = 0.0

        threshold = args.threshold
        n_above = sum(1 for x in history if x >= threshold)

        tem_drone = n_above >= min_windows_above

        print(
            f" p_atual={p:.3f}  |  p_max={p_max:.3f}  |  p_med={p_med:.3f}  "
            f"|  n_above={n_above} "
            f"=> {'🎯 DRONE DETECTADO' if tem_drone else '---'}"
        )

    print("🎙️  Escutando... (Ctrl + C para parar)\n")
    nn.start(on_res)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
    finally:
        nn.stop()


if __name__ == "__main__":
    main()
