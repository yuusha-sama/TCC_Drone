#!/usr/bin/env python3
"""
fusion_acoustic_rf.py — Fusão Áudio (Keras) + RF, com histerese e suavização.

Requisitos (pip):
  pip install numpy sounddevice tensorflow

Arquivos do projeto usados:
  - acoustic_models/adapter.py       (Keras-only)
  - acoustic_models/api.py
  - acoustic_models/features.py
  - acoustic_models/keras_backend.py

Como executar (exemplos):
  # Só acoustic (RF desativado):
  python fusion_acoustic_rf.py --model models/audio_classifier.keras --threshold 0.6

  # Com RF (ex.: p_rf constante 0.3):
  python fusion_acoustic_rf.py --model models/audio_classifier.keras --rf-const 0.3
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import argparse
import threading
import time
import numpy as np

from acoustic_models.adapter import AcousticNNModel


# =========================
# Configuração de Fusão
# =========================

@dataclass
class FusionConfig:
    # Pesos da fusão (p_total = w_audio*p_audio + w_rf*p_rf)
    w_audio: float = 0.6
    w_rf: float = 0.4

    # Limiar de decisão sobre a probabilidade combinada
    threshold: float = 0.5

    # Suavização exponencial (0 = sem suavização; 1 = mantém valor anterior)
    ema_alpha: float = 0.2

    # Histerese: exige N confirmações consecutivas para ligar/desligar estado "DRONE"
    min_confirm_on: int = 2
    min_confirm_off: int = 2


# =========================
# Estado Compartilhado
# =========================

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._p_audio: float = 0.0
        self._p_rf: float = 0.0
        self._last_total: float = 0.0

    def set_audio(self, p: float):
        with self._lock:
            self._p_audio = float(np.clip(p, 0.0, 1.0))

    def set_rf(self, p: float):
        with self._lock:
            self._p_rf = float(np.clip(p, 0.0, 1.0))

    def get(self):
        with self._lock:
            return self._p_audio, self._p_rf, self._last_total

    def set_total(self, p_total: float):
        with self._lock:
            self._last_total = float(np.clip(p_total, 0.0, 1.0))


# =========================
# Nó de Áudio (Keras-only)
# =========================

class AcousticNode:
    """
    Encapsula o AcousticNNModel (Keras) e alimenta o SharedState com p_audio.
    """
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str],
        n_fft: int,
        n_mels: int,
        sample_rate: int,
        win_s: float,
        hop_s: float,
        idx_drone: Optional[int],
        device_index: Optional[int],
        threshold_print: float = 0.5,
    ):
        self.state = None  # será atribuído depois (SharedState)
        # >>> KERAS-ONLY: não passar backend_kind <<<
        self.nn = AcousticNNModel(
            model_path=model_path,
            labels_path=labels_path,
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            win_s=win_s,
            hop_s=hop_s,
            idx_drone=idx_drone,
            device_index=device_index,
            threshold=threshold_print,
        )

    def attach_state(self, state: SharedState):
        self.state = state

    def _on_result(self, res: dict):
        if self.state is None:
            return
        p = float(res.get("p_drone", 0.0))
        self.state.set_audio(p)
        print(f"[AUDIO] p(drone)={p:.3f}")

    def start(self):
        if self.state is None:
            raise RuntimeError("Chame attach_state(state) antes de start()")
        self.nn.start(self._on_result)

    def stop(self):
        self.nn.stop()


# =========================
# Nó de RF (exemplo)
# =========================

class RFNode:
    def __init__(self, provider: Callable[[], float], period_s: float = 0.5):
        self.state: Optional[SharedState] = None
        self._provider = provider
        self._period_s = period_s
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def attach_state(self, state: SharedState):
        self.state = state

    def _poll_rf(self) -> float:
        return float(np.clip(self._provider(), 0.0, 1.0))

    def _worker(self):
        while not self._stop.is_set():
            p_rf = self._poll_rf()
            if self.state is not None:
                self.state.set_rf(p_rf)
            print(f"[RF]    p(drone)={p_rf:.3f}")
            time.sleep(self._period_s)

    def start(self):
        if self.state is None:
            raise RuntimeError("Chame attach_state(state) antes de start()")
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
            self._thr = None


# =========================
# Motor de Fusão
# =========================

class FusionEngine:
    def __init__(self, state: SharedState, cfg: FusionConfig, hold_on_s: float = 1.5):
        self.state = state
        self.cfg = cfg
        self._ema_total: Optional[float] = None
        self._is_drone: bool = False
        self._cnt_on = 0
        self._cnt_off = 0
        self._hold_on_s = float(hold_on_s)
        self._hold_until: float = 0.0

    def step(self):
        p_audio, p_rf, _ = self.state.get()
        p_total = self.cfg.w_audio * p_audio + self.cfg.w_rf * p_rf

        # suavização total
        if self._ema_total is None:
            self._ema_total = p_total
        else:
            a = float(np.clip(self.cfg.ema_alpha, 0.0, 1.0))
            self._ema_total = a * self._ema_total + (1.0 - a) * p_total

        self.state.set_total(self._ema_total)

        now = time.time()

        # histerese + hold
        if self._ema_total >= self.cfg.threshold:
            self._cnt_on += 1
            self._cnt_off = 0
            if not self._is_drone and self._cnt_on >= self.cfg.min_confirm_on:
                self._is_drone = True
                self._hold_until = now + self._hold_on_s
                self.on_event(True, self._ema_total)
            elif self._is_drone:
                # renova o hold enquanto estiver acima
                self._hold_until = now + self._hold_on_s
        else:
            self._cnt_off += 1
            self._cnt_on = 0
            if self._is_drone:
                # só desliga se passou do hold e acumulou confirmações de off
                if now >= self._hold_until and self._cnt_off >= self.cfg.min_confirm_off:
                    self._is_drone = False
                    self.on_event(False, self._ema_total)

        print(f"[FUSION] p_audio={p_audio:.3f}  p_rf={p_rf:.3f}  p_total(EMA)={self._ema_total:.3f}  => {'DRONE' if self._is_drone else '---'}")


# =========================
# CLI / Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Fusão Áudio(Keras) + RF com histerese")
    # Parâmetros do modelo Keras
    ap.add_argument("--model", required=True, help="Caminho do modelo Keras (.keras/.h5)")
    ap.add_argument("--labels", default=None, help="Caminho opcional para labels.txt")
    ap.add_argument("--idx_drone", type=int, default=None, help="Índice da classe 'drone' (se multiclasse). Se omitir, tenta inferir/assume 1")
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--rate", type=int, default=16000, help="Taxa de amostragem alvo")
    ap.add_argument("--win", type=float, default=1.0, help="Janela de áudio (s)")
    ap.add_argument("--hop", type=float, default=0.5, help="Passo (s)")
    ap.add_argument("--audio-device", type=int, default=None, help="Índice do microfone (sounddevice)")

    # RF (exemplos)
    ap.add_argument("--rf-const", type=float, default=None, help="Probabilidade RF constante [0..1]")
    ap.add_argument("--rf-sine", action="store_true", help="Gera p_rf ~ seno(t) só para teste")
    ap.add_argument("--rf-period", type=float, default=0.5, help="Período de atualização do RF (s)")

    # Fusão / decisão
    ap.add_argument("--w-audio", type=float, default=0.6)
    ap.add_argument("--w-rf", type=float, default=0.4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ema-alpha", type=float, default=0.2)
    ap.add_argument("--min-on", type=int, default=2)
    ap.add_argument("--min-off", type=int, default=2)

    args = ap.parse_args()

    state = SharedState()

    # Nó de áudio KERAS-ONLY
    audio_node = AcousticNode(
        model_path=args.model,
        labels_path=args.labels,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        sample_rate=args.rate,
        win_s=args.win,
        hop_s=args.hop,
        idx_drone=args.idx_drone,
        device_index=args.audio_device,
        threshold_print=args.threshold,
    )
    audio_node.attach_state(state)
    audio_node.start()

    # Nó de RF
    if args.rf_const is not None:
        const_val = float(np.clip(args.rf_const, 0.0, 1.0))
        provider = (lambda: const_val)
    elif args.rf_sine:
        t0 = time.time()
        def provider():
            t = time.time() - t0
            return 0.5 + 0.5 * np.sin(2*np.pi * (1/10.0) * t)
    else:
        provider = (lambda: 0.0)

    rf_node = RFNode(provider=provider, period_s=args.rf_period)
    rf_node.attach_state(state)
    rf_node.start()

    cfg = FusionConfig(
        w_audio=args.w_audio,
        w_rf=args.w_rf,
        threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        min_confirm_on=args.min_on,
        min_confirm_off=args.min_off,
    )
    engine = FusionEngine(state, cfg)

    print(">> Rodando fusão. Ctrl+C para sair.")
    try:
        while True:
            engine.step()
            time.sleep(args.hop)
    except KeyboardInterrupt:
        pass
    finally:
        print("Encerrando...")
        try:
            rf_node.stop()
        except Exception:
            pass
        try:
            audio_node.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
