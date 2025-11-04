from __future__ import annotations
from typing import Callable, Optional, Dict, List
import threading
import queue
import os
import numpy as np
import sounddevice as sd

from .api import load_backend, classify_window  # uses features + keras backend

class AcousticNNModel:
    """Adaptador para modelo acústico pré-treinado (Keras)."""
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        n_fft: int = 1024,
        n_mels: int = 64,
        sample_rate: int = 16000,
        win_s: float = 1.0,
        hop_s: float = 0.5,
        idx_drone: Optional[int] = None,
        device_index: Optional[int] = None,
        threshold: float = 0.5,
    ):
        self.backend = load_backend("keras", model_path)
        self.labels = self._load_labels(labels_path)
        self.idx_drone = idx_drone
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_s = win_s
        self.hop_s = hop_s
        self.device_index = device_index
        self.threshold = threshold

        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._buf = np.zeros(int(self.sample_rate * self.win_s), dtype=np.float32)
        self._stream: Optional[sd.InputStream] = None

    @staticmethod
    def _load_labels(path: Optional[str]) -> Optional[List[str]]:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            return lines or None
        return None

    def get_drone_index(self) -> int:
        if self.idx_drone is not None:
            return int(self.idx_drone)
        if self.labels:
            for i, name in enumerate(self.labels):
                if name.lower() == "drone":
                    return i
        return 1  # padrão comum: [background, drone]

    def classify_window(self, pcm: np.ndarray, sr: int) -> Dict[str, float]:
        idx_drone = self.get_drone_index()
        return classify_window(
            pcm, sr, backend=self.backend,
            n_fft=self.n_fft, n_mels=self.n_mels, idx_drone=idx_drone
        )

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(status)
        samples = indata.copy().squeeze()
        self._q.put(samples)

    def _worker(self, on_result: Callable[[Dict[str, float]], None]):
        hop_len = int(self.sample_rate * self.hop_s)
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            if len(chunk) >= hop_len:
                self._buf[:-hop_len] = self._buf[hop_len:]
                self._buf[-hop_len:] = chunk[:hop_len]
            else:
                self._buf = np.roll(self._buf, -len(chunk))
                self._buf[-len(chunk):] = chunk

            res = self.classify_window(self._buf, sr=self.sample_rate)
            on_result(res)

    def start(self, on_result: Callable[[Dict[str, float]], None]) -> None:
        if self._thr is not None:
            raise RuntimeError("Já iniciado")
        self._stop.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            callback=self._audio_cb, device=self.device_index
        )
        self._stream.start()
        self._thr = threading.Thread(target=self._worker, args=(on_result,), daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
            self._thr = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
