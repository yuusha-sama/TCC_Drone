#!/usr/bin/env python3
"""
acoustic_mfcc_classes_nolibrosa.py — MFCC sem librosa/numba (compatível Python 3.14)

Componentes:
  - MFCCConfig, MFCCExtractorNL (implementa MFCC + Δ + ΔΔ via NumPy/SciPy)
  - WAVRecorder, WAVDataset (iguais à versão anterior)
  - ModelTrainer (SVM + StandardScaler)
  - LiveAudioClassifier (tempo real com votação)

Dependências somente: numpy, scipy, soundfile, sounddevice, pandas, scikit-learn, joblib

Uso rápido está nos comentários ao final.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import queue
import time
import sys

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _SD_ERR = e

from scipy.fft import rfft
from scipy.signal import get_window
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# =========================
# Utils DSP (MFCC puro)
# =========================

def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    return 2595.0 * np.log10(1.0 + f/700.0)

def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    return 700.0 * (10**(m/2595.0) - 1.0)

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    # número de bins úteis na rFFT
    n_bins = n_fft//2 + 1
    # pontos em mel
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels+2)
    hz = mel_to_hz(mels)
    # mapeia para bins de frequência
    bins = np.floor((n_fft+1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(1, n_mels+1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        # subida
        for j in range(left, center):
            if 0 <= j < n_bins:
                fb[i-1, j] = (j - left) / max(1, (center - left))
        # descida
        for j in range(center, right):
            if 0 <= j < n_bins:
                fb[i-1, j] = (right - j) / max(1, (right - center))
    # normalização energética (Slaney-like)
    fb = fb / (np.sum(fb, axis=1, keepdims=True) + 1e-12)
    return fb

from scipy.fftpack import dct

def mfcc_from_signal(y: np.ndarray, sr: int, n_mfcc: int=20, n_fft: int=1024, hop_length: int=512,
                      n_mels: int=40, fmin: float=50.0, fmax: float=4000.0, pre_emph: float=0.97) -> np.ndarray:
    # mono + normalização leve
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx
    # pré-ênfase
    y[1:] = y[1:] - pre_emph * y[:-1]
    # framing com hop
    win = get_window('hann', n_fft, fftbins=True).astype(np.float32)
    n_frames = 1 + max(0, (len(y) - n_fft) // hop_length)
    if n_frames <= 0:
        # pad até ter 1 frame
        pad = np.zeros(n_fft - len(y) + 1, dtype=np.float32)
        y = np.concatenate([y, pad])
        n_frames = 1
    # filtro mel
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    S = np.empty((n_mels, n_frames), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_length
        frame = y[s:s+n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft-len(frame)))
        X = np.abs(rfft(frame * win))**2  # potência
        melE = fb @ X[:n_fft//2 + 1]
        S[:, i] = np.log(melE + 1e-12)
    # DCT-II ao longo de mels → MFCCs
    C = dct(S, type=2, axis=0, norm='ortho')  # shape: (n_mels, T)
    C = C[:n_mfcc, :]  # pega os primeiros coeficientes
    return C  # (n_mfcc, T)

def deltas(M: np.ndarray, order: int = 1, width: int = 9) -> np.ndarray:
    # derivada temporal simples (Janela ímpar)
    assert width % 2 == 1
    half = width//2
    denom = 2 * sum([i*i for i in range(1, half+1)])
    out = np.zeros_like(M)
    for t in range(M.shape[1]):
        acc = np.zeros(M.shape[0], dtype=np.float32)
        for n in range(1, half+1):
            t1 = min(M.shape[1]-1, t+n)
            t2 = max(0, t-n)
            acc += n * (M[:, t1] - M[:, t2])
        out[:, t] = acc / (denom + 1e-12)
    if order == 2:
        return deltas(out, order=1, width=width)
    return out

# =========================
# Config & Extractor (no-librosa)
# =========================
@dataclass
class MFCCConfig:
    sr: int = 16_000
    n_mfcc: int = 20
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 40
    fmin: float = 50.0
    fmax: float = 4000.0
    use_deltas: bool = True
    stats: str = "meanstd"  # 'meanstd' ou 'mean'

class MFCCExtractorNL:
    def __init__(self, cfg: MFCCConfig):
        self.cfg = cfg
    def _matrix(self, y: np.ndarray) -> np.ndarray:
        C = mfcc_from_signal(y, sr=self.cfg.sr, n_mfcc=self.cfg.n_mfcc, n_fft=self.cfg.n_fft,
                             hop_length=self.cfg.hop_length, n_mels=self.cfg.n_mels,
                             fmin=self.cfg.fmin, fmax=self.cfg.fmax)
        mats = [C]
        if self.cfg.use_deltas:
            d1 = deltas(C, order=1)
            d2 = deltas(C, order=2)
            mats += [d1, d2]
        return np.vstack(mats)  # (n_feats, T)
    def vectorize(self, y: np.ndarray) -> np.ndarray:
        F = self._matrix(y)
        if self.cfg.stats == "meanstd":
            mu = np.mean(F, axis=1)
            sd = np.std(F, axis=1)
            v = np.concatenate([mu, sd])
        else:
            v = np.mean(F, axis=1)
        return v.astype(np.float32)
    def wav_to_vector(self, wav_path: Path) -> np.ndarray:
        y, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
        if sr != self.cfg.sr:
            # simples reamostragem linear (para evitar dependência externa); para melhor qualidade, use resampy/samplerate
            ratio = self.cfg.sr / sr
            x_idx = np.arange(0, len(y))
            t_idx = np.arange(0, len(y)*ratio, 1.0)
            y = np.interp(t_idx, x_idx, y if y.ndim==1 else y.mean(axis=1)).astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return self.vectorize(y)

# =========================
# Coleta (gravação)
# =========================
class WAVRecorder:
    def __init__(self, sr: int = 16_000, device: int | None = None):
        if sd is None:
            raise RuntimeError(f"sounddevice indisponível: {_SD_ERR}")
        self.sr = sr
        self.device = device
    def record_chunks(self, label: str, outdir: Path, minutes: float = 1.0, chunk_sec: float = 5.0):
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        total = int(minutes*60*self.sr)
        chunk_samples = int(chunk_sec*self.sr)
        q = queue.Queue()
        def cb(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")
            q.put(indata.copy())
        print(f"[record] '{label}' {minutes:.1f} min, chunks {chunk_sec:.1f}s @ {self.sr} Hz → {outdir}")
        with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', callback=cb, device=self.device):
            buf = np.zeros(chunk_samples, dtype=np.float32)
            filled = 0; got = 0; idx = 0; t0 = int(time.time())
            while got < total:
                try:
                    x = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if x.ndim == 2:
                    x = x.mean(axis=1)
                m = min(len(x), chunk_samples-filled)
                buf[filled:filled+m] = x[:m]
                filled += m; got += m
                if filled >= chunk_samples:
                    fn = outdir / f"{label}_{t0}_{idx:04d}.wav"
                    sf.write(str(fn), buf, self.sr)
                    print("  [+]", fn.name)
                    idx += 1; filled = 0
        print("[record] ok")

# =========================
# Dataset
# =========================
class WAVDataset:
    def __init__(self, root: Path, extractor: MFCCExtractorNL):
        self.root = Path(root)
        self.ext = extractor
    def iter_wavs(self):
        for label_dir in sorted(self.root.glob('*')):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for wav in sorted(label_dir.glob('*.wav')):
                yield wav, label
    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for wav, label in self.iter_wavs():
            try:
                vec = self.ext.wav_to_vector(wav)
                row = {f"f{i}": float(v) for i, v in enumerate(vec)}
                row['label'] = label
                row['file'] = str(wav)
                rows.append(row)
            except Exception as e:
                print(f"[dataset] erro {wav}: {e}")
        return pd.DataFrame(rows)
    def to_csv(self, out_csv: Path) -> Path:
        df = self.to_dataframe()
        out_csv = Path(out_csv)
        df.to_csv(out_csv, index=False)
        print(f"[dataset] salvo → {out_csv}  ({len(df)} amostras, {len([c for c in df.columns if c.startswith('f')])} feats)")
        return out_csv

# =========================
# Treino / Modelo
# =========================
class ModelTrainer:
    def __init__(self, model=None):
        self.model = model or SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
        self.scaler = StandardScaler()
    def _split(self, X, y, test_size=0.2):
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    def fit(self, df: pd.DataFrame, test_size: float = 0.2, kfold: int = 0):
        feat_cols = [c for c in df.columns if c.startswith('f')]
        X = df[feat_cols].values.astype(np.float32)
        y = df['label'].values
        Xs = self.scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = self._split(Xs, y, test_size=test_size)
        self.model.fit(Xtr, ytr)
        yhat = self.model.predict(Xte)
        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat, average='weighted')
        rep = classification_report(yte, yhat)
        cm = confusion_matrix(yte, yhat)
        cv_info = None
        if kfold and kfold > 1:
            cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, Xs, y, cv=cv, scoring='f1_weighted')
            cv_info = (scores.mean(), scores.std())
        return {"acc": acc, "f1": f1, "report": rep, "cm": cm, "cv": cv_info}
    def save(self, scaler_path: Path, model_path: Path):
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.model, model_path)
        print(f"[model] scaler → {scaler_path}\n[model] model  → {model_path}")
    @staticmethod
    def load(scaler_path: Path, model_path: Path) -> 'ModelTrainer':
        mt = ModelTrainer()
        mt.scaler = joblib.load(scaler_path)
        mt.model = joblib.load(model_path)
        return mt
    def predict_proba(self, X_vec: np.ndarray) -> tuple[str, float]:
        Xs = self.scaler.transform(X_vec.reshape(1, -1))
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(Xs)[0]
            idx = int(np.argmax(proba))
            return self.model.classes_[idx], float(proba[idx])
        else:
            y = self.model.predict(Xs)[0]
            return y, 1.0

# =========================
# Live
# =========================
class LiveAudioClassifier:
    def __init__(self, trainer: ModelTrainer, extractor: MFCCExtractorNL, frame_sec: float = 1.0, hop_sec: float = 0.25, device: int | None = None):
        if sd is None:
            raise RuntimeError(f"sounddevice indisponível: {_SD_ERR}")
        self.trainer = trainer
        self.ext = extractor
        self.sr = extractor.cfg.sr
        self.frame = int(frame_sec * self.sr)
        self.hop = int(hop_sec * self.sr)
        self.device = device
    def run(self, vote: int = 4):
        q = queue.Queue()
        def cb(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")
            q.put(indata.copy())
        print(f"[live] sr={self.sr} frame={self.frame/self.sr:.2f}s hop={self.hop/self.sr:.2f}s — Ctrl+C para sair")
        with sd.InputStream(samplerate=self.sr, channels=1, dtype='float32', callback=cb, blocksize=self.hop, device=self.device):
            buf = np.zeros(self.frame, dtype=np.float32)
            filled = 0
            history: list[str] = []
            try:
                while True:
                    try:
                        x = q.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if x.ndim == 2:
                        x = x.mean(axis=1)
                    m = min(len(x), self.frame - filled)
                    buf[filled:filled+m] = x[:m]
                    filled += m
                    if filled >= self.frame:
                        vec = self.ext.vectorize(buf)
                        label, conf = self.trainer.predict_proba(vec)
                        history.append(label)
                        if len(history) > vote:
                            history = history[-vote:]
                        labels, counts = np.unique(history, return_counts=True)
                        maj = labels[np.argmax(counts)]
                        majp = counts.max() / len(history)
                        print(f"[live] pred={label} conf={conf:.2f} | vote={maj} ({majp:.2f})")
                        buf[:-self.hop] = buf[self.hop:]
                        filled = self.frame - self.hop
            except KeyboardInterrupt:
                print("[live] encerrado")

# =========================
# Exemplo de uso (colar/rodar conforme necessidade)
# =========================
if __name__ == "__main__":
    # 1) Coleta
    # rec = WAVRecorder(sr=16000)
    # rec.record_chunks('other', Path('data/other'), minutes=1.0, chunk_sec=5.0)
    # rec.record_chunks('drone', Path('data/drone'), minutes=1.0, chunk_sec=5.0)

    # 2) Dataset
    cfg = MFCCConfig(sr=16000, n_mfcc=20)
    ext = MFCCExtractorNL(cfg)
    ds = WAVDataset(Path('data'), ext)
    csv_path = ds.to_csv(Path('dataset.csv'))

    # 3) Treino
    df = pd.read_csv(csv_path)
    trainer = ModelTrainer()
    metrics = trainer.fit(df, test_size=0.2, kfold=5)
    print("[metrics] acc=", metrics['acc'], "f1=", metrics['f1'])
    print(metrics['report'])
    print(metrics['cm'])
    trainer.save(Path('scaler.pkl'), Path('model.pkl'))

    # 4) Live
    # trainer2 = ModelTrainer.load(Path('scaler.pkl'), Path('model.pkl'))
    # live = LiveAudioClassifier(trainer2, ext, frame_sec=1.0, hop_sec=0.25)
    # live.run(vote=4)
