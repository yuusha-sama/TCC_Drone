#!/usr/bin/env python3
from __future__ import annotations
import os, argparse, pathlib, numpy as np, soundfile as sf
from sklearn.model_selection import train_test_split
from acoustic_models.features import ensure_mono_16k, logmel_spectrogram
import tensorflow as tf
from tensorflow.keras import layers as L, models as M, optimizers as O, callbacks as CB

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}

def list_audio_files(root: str):
    items = []
    rootp = pathlib.Path(root)
    if not rootp.exists(): raise FileNotFoundError(f"Pasta não encontrada: {root}")
    for cls_dir in sorted([p for p in rootp.iterdir() if p.is_dir()]):
        cls = cls_dir.name
        for fp in cls_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in AUDIO_EXTS:
                items.append((str(fp), cls))
    if not items: raise RuntimeError(f"Nenhum áudio em {root}")
    return items

def load_audio(path: str):
    x, sr = sf.read(path, always_2d=False)
    x, sr = ensure_mono_16k(x, sr)
    return x.astype(np.float32), sr

def wav_to_logmel(x: np.ndarray, sr: int, n_fft=1024, n_mels=64):
    LM = logmel_spectrogram(x, sr, n_fft=n_fft, hop_length=n_fft//4, n_mels=n_mels)
    LM = (LM - LM.mean()) / (LM.std() + 1e-6)
    return LM.astype(np.float32)

def pad_or_crop_time(LM: np.ndarray, t_frames: int):
    T, F = LM.shape
    if T == t_frames: return LM
    if T > t_frames:
        s = (T - t_frames)//2
        return LM[s:s+t_frames, :]
    out = np.zeros((t_frames, F), dtype=LM.dtype)
    out[:min(T, t_frames)] = LM[:min(T, t_frames)]
    return out

def build_cnn(t_frames: int, n_mels: int, n_classes: int):
    inp = L.Input(shape=(t_frames, n_mels, 1), name="logmel")
    x = L.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = L.BatchNormalization()(x); x = L.MaxPool2D(2)(x)
    x = L.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x); x = L.MaxPool2D(2)(x)
    x = L.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = L.BatchNormalization()(x); x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.25)(x)
    if n_classes == 1:
        out = L.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"; metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    else:
        out = L.Dense(n_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"; metrics = ["accuracy"]
    model = M.Model(inp, out)
    model.compile(optimizer=O.Adam(1e-3), loss=loss, metrics=metrics)
    return model

def make_label_map(class_names, binary: bool):
    if binary:
        pos_names = {"drone", "uav", "quad", "quadcopter"}
        label_map = {c:(1 if c.lower() in pos_names else 0) for c in class_names}
        labels_txt = ["background", "drone"]
    else:
        sorted_classes = sorted(class_names)
        label_map = {c:i for i,c in enumerate(sorted_classes)}
        labels_txt = sorted_classes
    return label_map, labels_txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Ex.: ...\\Binary_Drone_Audio")
    ap.add_argument("--binary", action="store_true", help="Drone vs background")
    ap.add_argument("--multiclass", dest="binary", action="store_false", help="Usa todas as pastas como classes")
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--t_frames", type=int, default=63)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save_dir", default="models")
    ap.add_argument("--save_tflite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    items = list_audio_files(args.data_root)
    class_names = sorted({c for _,c in items})
    label_map, labels_txt = make_label_map(class_names, binary=args.binary)

    X, y = [], []
    for path, cls in items:
        try:
            wav, sr = load_audio(path)
            LM  = wav_to_logmel(wav, sr, n_fft=args.n_fft, n_mels=args.n_mels)
            LM  = pad_or_crop_time(LM, args.t_frames)
            X.append(LM[..., None]); y.append(label_map[cls])
        except Exception as e:
            print(f"[WARN] pulando {path}: {e}")

    X = np.stack(X, axis=0).astype(np.float32)  # [N,T,F,1]
    y = np.array(y, dtype=np.int64)

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=args.val_split, random_state=42, stratify=y)

    n_classes = 1 if (args.binary and len(set(y)) <= 2) else len(set(y))
    model = build_cnn(args.t_frames, args.n_mels, n_classes)

    ckpt = CB.ModelCheckpoint(os.path.join(args.save_dir, "audio_classifier.keras"),
                              monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    es = CB.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True)

    model.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=args.epochs, batch_size=args.batch, callbacks=[ckpt, es])

    model.save(os.path.join(args.save_dir, "audio_classifier.h5"))
    with open(os.path.join(args.save_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for n in (["background","drone"] if args.binary else sorted(class_names)):
            f.write(n+"\n")

    if args.save_tflite:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        tfl = conv.convert()
        with open(os.path.join(args.save_dir, "audio_classifier.tflite"), "wb") as f:
            f.write(tfl)

    print("OK. Modelos salvos em", args.save_dir)

if __name__ == "__main__":
    main()
