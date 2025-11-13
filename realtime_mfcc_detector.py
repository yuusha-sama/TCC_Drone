# realtime_mfcc_detector.py
import numpy as np
import sounddevice as sd
import joblib
from collections import deque

from DB.audio_mfcc_utils import extract_mfcc_from_buffer

# Mesmo sample rate do treino
SAMPLE_RATE = 16000
BLOCK_DURATION = 1.0           # segundos analisados por vez
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

MODEL_PATH = "models/drone_mfcc_rf.pkl"
clf = joblib.load(MODEL_PATH)

DRONE_LABEL = "Drone"

# ===== AJUSTES DE DECISÃO =====
# agora limite de ~30% pra detectar
DRONE_AVG_THRESHOLD = 0.30     # média das últimas probabilidades
DRONE_STRONG_THRESHOLD = 0.60  # prob atual bem alta já força detecção
PROB_HISTORY_SIZE = 5          # quantos blocos entram na média

prob_history = deque(maxlen=PROB_HISTORY_SIZE)


def callback(indata, frames, time, status):
    if status:
        print(status)

    # indata: [frames, channels] -> mono
    audio_mono = indata.mean(axis=1)

    # Extrai MFCC do buffer atual
    features = extract_mfcc_from_buffer(
        audio_mono,
        sr=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=1024,
        hop_length=512,
    )

    X = features.reshape(1, -1)

    # Probabilidade da classe "Drone"
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        classes = clf.classes_
        if DRONE_LABEL in classes:
            idx = list(classes).index(DRONE_LABEL)
            drone_prob = float(proba[idx])
        else:
            drone_prob = 0.0
    else:
        pred = clf.predict(X)[0]
        drone_prob = 1.0 if pred == DRONE_LABEL else 0.0

    # Atualiza histórico
    prob_history.append(drone_prob)
    avg_prob = sum(prob_history) / len(prob_history)

    # Decisão: média >= 0.30 OU prob atual >= 0.60
    if avg_prob >= DRONE_AVG_THRESHOLD or drone_prob >= DRONE_STRONG_THRESHOLD:
        print(
            f"🚁 Drone detectado! "
            f"(prob média: {avg_prob:.2f} | prob atual: {drone_prob:.2f})"
        )
    else:
        print(
            f"Sem drone "
            f"(prob média: {avg_prob:.2f} | prob atual: {drone_prob:.2f})"
        )


def main():
    print("Iniciando detecção em tempo real com MFCC...")
    print("Pressione Ctrl+C para parar.")

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    main()
