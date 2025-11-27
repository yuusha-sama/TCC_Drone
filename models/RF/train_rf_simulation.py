from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


# Pastas de apoio
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]

CSV_PATH = THIS_DIR / "dataset_rf_simulado.csv"
MODEL_PATH = ROOT_DIR / "drone_rf_regressor.pkl"


def gerar_dataset_rf(
    n_amostras: int = 5000,
    distancia_max: float = 300.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera um dataset sintético de RF.
    A distância cresce de 0 até distancia_max.
    O sinal cai logaritmicamente com ruído gaussiano.
    Isso aqui é só para simulação, não é medida real.
    """
    rng = np.random.default_rng(seed)

    # Distância "real" que queremos recuperar no modelo
    distance_real = rng.uniform(0.0, distancia_max, size=n_amostras)

    # RSSI bruto em dB: curva simples de perda em espaço livre + ruído
    rssi_raw = -20.0 - 28.0 * np.log10(distance_real + 1.0) + rng.normal(0.0, 2.0, size=n_amostras)

    # SNR derivado do RSSI, só para ter uma feature com escala mais amigável
    snr_raw = (rssi_raw + 105.0) / 3.0 + rng.normal(0.0, 1.0, size=n_amostras)

    df = pd.DataFrame(
        {
            "distance_real": distance_real,
            "rssi_raw": rssi_raw,
            "snr_raw": snr_raw,
        }
    )

    return df


def treinar_regressor_rf(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Treina um RandomForestRegressor usando o dataset sintético.
    Aqui a ideia é o modelo aprender a relação distância x sinal
    que foi imposta pela simulação.
    """

    # Se quiser usar RSSI + SNR, é só colocar as duas colunas aqui.
    features = ["snr_raw"]
    target = "distance_real"

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    reg.fit(X_train, y_train)

    # Avaliação rápida só para ter ideia do erro
    y_pred = reg.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print("===== TREINO REGRESSOR RF (SIMULAÇÃO) =====")
    print(f"MAE   (erro médio absoluto): {mae:.3f}")
    print(f"RMSE (raiz do erro quadrático médio): {rmse:.3f}")
    print(f"R²    (coeficiente de determinação): {r2:.4f}")
    print("============================================")

    return reg


def main():
    # Gera dataset sintético
    df = gerar_dataset_rf(
        n_amostras=5000,
        distancia_max=300.0,
        seed=42,
    )

    # Salva o CSV para análise e para a etapa de avaliação
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Dataset sintético salvo em: {CSV_PATH}")

    # Treina o regressor em cima desse dataset
    regressor = treinar_regressor_rf(df)

    # Salva o modelo na raiz de models/
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, MODEL_PATH)
    print(f"Modelo de regressão RF salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
