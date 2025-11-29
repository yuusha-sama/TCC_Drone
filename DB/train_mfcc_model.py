# DB/train_mfcc_model.py
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from kb_mongo_config import get_kb_collection
from audio_mfcc_utils import extract_mfcc_from_file


def carregar_dados():
    """
    Lê do Mongo os caminhos dos .wav, extrai MFCC e monta X, y.
    """
    collection = get_kb_collection()

    # Ajusta esses nomes de classe se no Mongo estiver diferente
    docs = list(
        collection.find(
            {"classe": {"$in": ["Drone", "unknown", "drone", "nao_drone"]}}
        )
    )

    X = []
    y = []

    for doc in docs:
        caminho_abs = Path(doc["caminho_absoluto"])
        classe = doc["classe"]

        if not caminho_abs.exists():
            print(f"⚠️ Arquivo não encontrado, pulando: {caminho_abs}")
            continue

        try:
            features = extract_mfcc_from_file(caminho_abs)
        except Exception as e:
            print(f"Erro extraindo MFCC de {caminho_abs}: {e}")
            continue

        X.append(features)

        # Normaliza o rótulo
        if classe.lower().startswith("drone"):
            y.append("Drone")
        else:
            y.append("Unknown")

    X = np.array(X)
    y = np.array(y)

    return X, y


def treinar_modelo():
    X, y = carregar_dados()

    print(f"Total de amostras: {len(y)}")
    classes, counts = np.unique(y, return_counts=True)
    print("Distribuição de classes:")
    for c, n in zip(classes, counts):
        print(f"  {c}: {n}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight={"Drone": 2.0, "Unknown": 1.0},  # <-- peso maior pra Drone
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n===== RELATÓRIO DE CLASSIFICAÇÃO =====")
    print(classification_report(y_test, y_pred))

    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "drone_mfcc_rf.pkl"
    joblib.dump(clf, model_path)
    print(f"\nModelo salvo em: {model_path}")


if __name__ == "__main__":
    treinar_modelo()
