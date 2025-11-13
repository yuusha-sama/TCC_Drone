#!/usr/bin/env python3
"""
rf_locator_classes.py — RF locator com RTL-SDR (V3/V4) + CLI

Mudanças focadas em estabilidade do RTL-SDR Blog V4:
- Retentativa com backoff para set_center_freq e read_samples (erros -9 PIPE e -1 I/O).
- retune() robusto que reabre o dispositivo entre tentativas.
- Opção de reabrir o SDR a cada hop no sweep (padrão: True).
- Parâmetros conservadores: 2.048 MSPS, buf_len=128k, settle=0.20s, hold=0.20s.
- Datetime timezone-aware.

Dependências:
    pip install pyrtlsdr numpy scipy pandas
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Driver
try:
    from rtlsdr import RtlSdr
except Exception as _err:
    RtlSdr = None
    _IMPORT_ERR = _err


# =========================
# Config
# =========================
@dataclass
class SDRConfig:
    samp_rate: float = 2.048e6
    gain: str | float = "auto"
    ppm: Optional[float] = None
    buf_len: int = 128_000
    settle: float = 0.20  # tempo após sintonia


# =========================
# Dispositivo
# =========================
class SDRDevice:
    def __init__(self, cfg: SDRConfig):
        if RtlSdr is None:
            raise RuntimeError(f"pyrtlsdr/rtlsdr não encontrado: {_IMPORT_ERR}")
        self.cfg = cfg
        self._sdr: Optional[RtlSdr] = None

    def open(self) -> None:
        self._sdr = RtlSdr()
        # Sample rate
        try:
            self._sdr.sample_rate = float(self.cfg.samp_rate)
        except Exception:
            pass
        # Modo tuner
        try:
            self._sdr.set_direct_sampling(0)
        except Exception:
            pass
        # Ganho
        try:
            if self.cfg.gain == "auto":
                try:
                    self._sdr.gain = "auto"
                except Exception:
                    pass  # deixa default
            else:
                self._sdr.gain = float(self.cfg.gain)
        except Exception:
            pass
        # PPM
        if self.cfg.ppm is not None:
            try:
                self._sdr.freq_correction = int(self.cfg.ppm)
            except Exception:
                pass

    def close(self) -> None:
        if self._sdr is not None:
            try:
                self._sdr.close()
            except Exception:
                pass
            self._sdr = None

    # ---- util: classifica erro de USB
    @staticmethod
    def _is_usb_glitch(exc: Exception) -> bool:
        msg = str(exc)
        return ("-9" in msg) or ("PIPE" in msg.upper()) or ("-1" in msg and "INPUT/OUTPUT" in msg.upper())

    # ---- leitura com retry (1x)
    def read_samples(self, n: Optional[int] = None) -> np.ndarray:
        assert self._sdr is not None, "SDR não aberto"
        n = int(n or self.cfg.buf_len)
        try:
            return self._sdr.read_samples(n)
        except Exception as e:
            if self._is_usb_glitch(e):
                # reabre e tenta de novo
                self.close()
                time.sleep(0.3)
                self.open()
                return self._sdr.read_samples(n)
            raise

    # ---- retune robusto
    def retune(self, freq_hz: float, attempts: int = 5, base_wait: float = 0.15) -> None:
        """
        Tenta sintonizar até N vezes. Entre tentativas, reabre o SDR e
        aplica backoff exponencial: base_wait * (i^1.2).
        """
        assert self._sdr is not None, "SDR não aberto"
        last_err = None
        for i in range(1, attempts + 1):
            try:
                self._sdr.center_freq = float(freq_hz)
                time.sleep(self.cfg.settle)
                return
            except Exception as e:
                last_err = e
                if not self._is_usb_glitch(e):
                    break
                # Reabrir e aguardar um pouco
                self.close()
                wait = base_wait * (i ** 1.2)
                time.sleep(wait)
                self.open()
        # Se chegou aqui, falhou
        raise last_err or RuntimeError("Falha ao sintonizar")

    # Compat: alias usado no resto do código
    def set_center_freq(self, freq_hz: float) -> None:
        self.retune(freq_hz)


# =========================
# Processamento
# =========================
class SpectrumProcessor:
    @staticmethod
    def welch_psd(samples: np.ndarray, nfft: int = 4096) -> np.ndarray:
        n = len(samples)
        if n <= 0:
            return np.zeros(nfft)
        win = np.hanning(n)
        X = np.fft.fft(samples * win, n=nfft)
        X = np.fft.fftshift(X)
        return 20.0 * np.log10(np.abs(X) + 1e-15)

    @staticmethod
    def freq_axis(center_hz: float, samp_rate: float, nfft: int) -> np.ndarray:
        return center_hz + np.linspace(-samp_rate / 2.0, samp_rate / 2.0, nfft, endpoint=False)

    @staticmethod
    def detect_peaks(freqs: np.ndarray, psd: np.ndarray, rel_thresh_db: float = 8.0,
                     min_spacing_hz: float = 25_000.0) -> List[Tuple[float, float]]:
        floor = np.median(psd)
        mask = psd >= (floor + rel_thresh_db)
        peaks: List[Tuple[float, float]] = []
        last_f = -1e99
        for f, p, m in zip(freqs, psd, mask):
            if not m:
                continue
            if (f - last_f) < min_spacing_hz:
                if peaks and p > peaks[-1][1]:
                    peaks[-1] = (f, p)
                continue
            peaks.append((f, p))
            last_f = f
        return peaks

    @staticmethod
    def top_n_peaks(freqs: np.ndarray, psd: np.ndarray, n: int = 8) -> List[Tuple[float, float]]:
        idx = np.argsort(psd)[-n:][::-1]
        return [(float(freqs[i]), float(psd[i])) for i in idx]


# =========================
# Funcionalidades
# =========================
class SanityCheck:
    def __init__(self, cfg: SDRConfig):
        self.cfg = cfg
        self.dev = SDRDevice(cfg)
        self.proc = SpectrumProcessor()

    def run(self, freq_center: Optional[float] = None, reads: int = 4, nfft: int = 4096,
            top_n: int = 8, csv_out: Optional[Path] = None) -> dict:
        self.dev.open()
        try:
            if freq_center is not None:
                self.dev.retune(freq_center)
            else:
                try:
                    _ = float(self.dev._sdr.center_freq)
                except Exception:
                    self.dev.retune(100e6)

            acc = None
            for _ in range(max(1, reads)):
                samples = self.dev.read_samples(self.cfg.buf_len)
                spec = self.proc.welch_psd(samples, nfft=nfft)
                acc = spec if acc is None else np.maximum(acc, spec)
                time.sleep(0.05)

            center = float(self.dev._sdr.center_freq)
            freqs = self.proc.freq_axis(center, self.cfg.samp_rate, nfft)
            floor = float(np.median(acc))
            tops = self.proc.top_n_peaks(freqs, acc, n=top_n)

            if csv_out is not None:
                with Path(csv_out).open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp_iso", "freq_hz", "psd_db"])
                    ts = datetime.now(timezone.utc).isoformat()
                    for fr, pv in zip(freqs, acc):
                        w.writerow([ts, int(fr), float(pv)])

            return {"center_freq": center, "floor_db": floor, "top_peaks": tops, "freqs": freqs, "psd": acc}
        finally:
            self.dev.close()


class Sweeper:
    def __init__(self, cfg: SDRConfig):
        self.cfg = cfg
        self.dev = SDRDevice(cfg)
        self.proc = SpectrumProcessor()

    def _make_centers(self, f_start: float, f_stop: float, step: float) -> np.ndarray:
        # Centros seguros: evitam sair da banda pedida
        # usamos centers que cobrem a faixa deslocando metade da SR em cada lado
        sr = float(self.cfg.samp_rate)
        first = f_start + sr / 2.0
        last = f_stop - sr / 2.0
        if last < first:
            return np.array([(f_start + f_stop) / 2.0])
        n = max(1, int(math.floor((last - first) / max(step, 1.0)) + 1))
        return first + np.arange(n) * step

    def sweep(self, f_start: float, f_stop: float, step: float, avg: int = 3,
              nfft: int = 4096, hold: float = 0.20, reopen_each_hop: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        if f_stop <= f_start:
            raise ValueError("f_stop deve ser maior que f_start")
        centers = self._make_centers(f_start, f_stop, step)

        # Estratégia: opcionalmente reabrir a cada hop (estável p/ V4)
        if reopen_each_hop:
            freqs_all, psd_all = [], []
            for i, fc in enumerate(centers):
                self.dev.open()
                try:
                    self.dev.retune(fc)
                    acc = None
                    for _ in range(max(1, avg)):
                        samples = self.dev.read_samples(self.cfg.buf_len)
                        spec = self.proc.welch_psd(samples, nfft=nfft)
                        acc = spec if acc is None else np.maximum(acc, spec)
                        time.sleep(hold)
                    freqs = self.proc.freq_axis(fc, self.cfg.samp_rate, nfft)
                    freqs_all.append(freqs)
                    psd_all.append(acc)
                    print(f"[sweep] {i+1}/{len(centers)}  fc={fc/1e6:.3f} MHz  ok")
                finally:
                    self.dev.close()
            freqs_concat = np.concatenate(freqs_all)
            psd_concat = np.concatenate(psd_all)
        else:
            self.dev.open()
            try:
                freqs_all, psd_all = [], []
                for i, fc in enumerate(centers):
                    self.dev.retune(fc)
                    acc = None
                    for _ in range(max(1, avg)):
                        samples = self.dev.read_samples(self.cfg.buf_len)
                        spec = self.proc.welch_psd(samples, nfft=nfft)
                        acc = spec if acc is None else np.maximum(acc, spec)
                        time.sleep(hold)
                    freqs = self.proc.freq_axis(fc, self.cfg.samp_rate, nfft)
                    freqs_all.append(freqs)
                    psd_all.append(acc)
                    print(f"[sweep] {i+1}/{len(centers)}  fc={fc/1e6:.3f} MHz  ok")
                freqs_concat = np.concatenate(freqs_all)
                psd_concat = np.concatenate(psd_all)
            finally:
                self.dev.close()

        peaks = self.proc.detect_peaks(freqs_concat, psd_concat)
        return freqs_concat, psd_concat, peaks

    @staticmethod
    def save_csv(path: Path, freqs: np.ndarray, psd: np.ndarray, peaks: List[Tuple[float, float]]):
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_iso", "freq_hz", "psd_db"])
            ts = datetime.now(timezone.utc).isoformat()
            for fr, pv in zip(freqs, psd):
                w.writerow([ts, int(fr), float(pv)])
            w.writerow([])
            w.writerow(["PEAKS_START"])
            w.writerow(["timestamp_iso", "peak_freq_hz", "peak_db"])
            for f0, p0 in peaks:
                w.writerow([ts, int(f0), float(p0)])


class Tracker:
    def __init__(self, cfg: SDRConfig):
        self.cfg = cfg
        self.dev = SDRDevice(cfg)
        self.proc = SpectrumProcessor()

    def track(self, freq_hz: float, avg: int = 4, nfft: int = 2048,
              out_csv: Optional[Path] = None, runtime_s: Optional[float] = None,
              sample_interval_s: float = 0.5):
        self.dev.open()
        fobj = None
        try:
            self.dev.retune(freq_hz)
            writer = None
            if out_csv is not None:
                fobj = Path(out_csv).open("w", newline="")
                writer = csv.writer(fobj)
                writer.writerow(["timestamp_iso", "freq_hz", "rssi_db"])
            start = time.time()
            while True:
                acc = None
                for _ in range(max(1, avg)):
                    samples = self.dev.read_samples(self.cfg.buf_len)
                    spec = self.proc.welch_psd(samples, nfft=nfft)
                    acc = spec if acc is None else np.maximum(acc, spec)
                freqs = self.proc.freq_axis(freq_hz, self.cfg.samp_rate, nfft)
                idx = int(np.argmin(np.abs(freqs - freq_hz)))
                rssi = float(acc[idx])
                ts = datetime.now(timezone.utc).isoformat()
                print(f"[track] {ts}  {freq_hz/1e6:.6f} MHz  {rssi:.1f} dBFS")
                if writer is not None:
                    writer.writerow([ts, int(freq_hz), rssi])
                    fobj.flush()
                if runtime_s is not None and (time.time() - start) > runtime_s:
                    break
                time.sleep(sample_interval_s)
        except KeyboardInterrupt:
            print("[track] interrompido pelo usuário")
        finally:
            if fobj is not None:
                fobj.close()
            self.dev.close()


class BearingAssistant:
    def __init__(self, cfg: SDRConfig):
        self.cfg = cfg
        self.dev = SDRDevice(cfg)
        self.proc = SpectrumProcessor()

    def run(self, freq_hz: float, avg: int = 6, nfft: int = 2048, note: str = "") -> dict:
        self.dev.open()
        try:
            self.dev.retune(freq_hz)
            measures = []
            print("Bearing: digite azimute (0–359). ENTER vazio finaliza.")
            while True:
                s = input("Azimute: ").strip()
                if s == "":
                    break
                try:
                    az = float(s) % 360.0
                except Exception:
                    print("Valor inválido.")
                    continue
                acc = None
                for _ in range(max(1, avg)):
                    samples = self.dev.read_samples(self.cfg.buf_len)
                    spec = self.proc.welch_psd(samples, nfft=nfft)
                    acc = spec if acc is None else np.maximum(acc, spec)
                freqs = self.proc.freq_axis(freq_hz, self.cfg.samp_rate, len(acc))
                idx = int(np.argmin(np.abs(freqs - freq_hz)))
                rssi = float(acc[idx])
                print(f"  medido {rssi:.1f} dB em {az:.1f}°")
                measures.append((az, rssi))
            if not measures:
                return {"measures": [], "az_est": None, "note": note}
            azs = np.array([m[0] for m in measures])
            rss = np.array([m[1] for m in measures])
            w = 10 ** (rss / 20.0)
            ang = np.deg2rad(azs)
            x = (w * np.cos(ang)).sum()
            y = (w * np.sin(ang)).sum()
            az_est = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
            return {"measures": measures, "az_est": float(az_est), "note": note}
        finally:
            self.dev.close()


# =========================
# CLI
# =========================
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RF locator (RTL-SDR) — sanity, sweep, track, bearing")
    p.add_argument("--samp-rate", type=float, default=SDRConfig.samp_rate)
    p.add_argument("--gain", type=str, default=str(SDRConfig.gain))
    p.add_argument("--ppm", type=float, default=None)
    p.add_argument("--buf-len", type=int, default=SDRConfig.buf_len)
    p.add_argument("--settle", type=float, default=SDRConfig.settle)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sanity", help="teste rápido do dongle + antena")
    s.add_argument("--freq-center", type=float, default=None)
    s.add_argument("--reads", type=int, default=4)
    s.add_argument("--nfft", type=int, default=4096)
    s.add_argument("--top", type=int, default=8)
    s.add_argument("--out", type=Path, default=None)

    s = sub.add_parser("sweep", help="varrer banda e detectar picos")
    s.add_argument("--f-start", type=float, required=True)
    s.add_argument("--f-stop", type=float, required=True)
    s.add_argument("--step", type=float, default=1.2e6)  # um pouco menor que 1 SR para estabilidade
    s.add_argument("--avg", type=int, default=3)
    s.add_argument("--nfft", type=int, default=4096)
    s.add_argument("--hold", type=float, default=0.20)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--reopen-each-hop", type=lambda x: str(x).lower() not in ["0","false","no","n"], default=True)

    s = sub.add_parser("track", help="monitorar uma frequência (RSSI vs tempo)")
    s.add_argument("--freq", type=float, required=True)
    s.add_argument("--avg", type=int, default=4)
    s.add_argument("--nfft", type=int, default=2048)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--runtime", type=float, default=None)
    s.add_argument("--interval", type=float, default=0.5)

    s = sub.add_parser("bearing", help="assistente manual de bearing")
    s.add_argument("--freq", type=float, required=True)
    s.add_argument("--avg", type=int, default=6)
    s.add_argument("--nfft", type=int, default=2048)
    s.add_argument("--note", type=str, default="")

    return p


def main(argv=None):
    args = build_cli().parse_args(argv)
    cfg = SDRConfig(
        samp_rate=args.samp_rate,
        gain=args.gain,
        ppm=args.ppm,
        buf_len=args.buf_len,
        settle=args.settle,
    )

    if args.cmd == "sanity":
        res = SanityCheck(cfg).run(
            freq_center=args.freq_center,
            reads=args.reads,
            nfft=args.nfft,
            top_n=args.top,
            csv_out=args.out,
        )
        print(f"Center: {res['center_freq']/1e6:.6f} MHz")
        print(f"Floor:  {res['floor_db']:.1f} dBFS")
        print("Top peaks:")
        for f, p in res["top_peaks"]:
            print(f"  {f/1e6:9.6f} MHz  {p:6.1f} dBFS")

    elif args.cmd == "sweep":
        freqs, psd, peaks = Sweeper(cfg).sweep(
            f_start=args.f_start,
            f_stop=args.f_stop,
            step=args.step,
            avg=args.avg,
            nfft=args.nfft,
            hold=args.hold,
            reopen_each_hop=args.reopen_each_hop,
        )
        Sweeper.save_csv(args.out, freqs, psd, peaks)
        print(f"[ok] sweep salvo em {args.out.resolve()} | picos={len(peaks)}")

    elif args.cmd == "track":
        Tracker(cfg).track(
            freq_hz=args.freq,
            avg=args.avg,
            nfft=args.nfft,
            out_csv=args.out,
            runtime_s=args.runtime,
            sample_interval_s=args.interval,
        )

    elif args.cmd == "bearing":
        res = BearingAssistant(cfg).run(
            freq_hz=args.freq,
            avg=args.avg,
            nfft=args.nfft,
            note=args.note,
        )
        print("Resultado bearing:", res)

if __name__ == "__main__":
    main()
