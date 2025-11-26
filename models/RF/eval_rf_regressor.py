from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pastas principais
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[1]

CSV_PATH = THIS_DIR / "dataset_rf_simulado.csv"
MODEL_PATH = ROOT_DIR / "drone_rf_regressor.pkl"
OUT_FIG = THIS_DIR / "real_vs_previsto_rf.png"


def main():
    # Lê o dataset gerado na simulação
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV de simulação não encontrado em: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Se os nomes de coluna forem outros, ajusta aqui
    features = ["snr_raw"]          # ou ["rssi_raw", "snr_raw"]
    target = "distance_real"

    if not set(features).issubset(df.columns):
        raise ValueError(f"Colunas de entrada não encontradas no CSV: {features}")
    if target not in df.columns:
        raise ValueError(f"Coluna de alvo '{target}' não encontrada no CSV.")

    X = df[features].values
    y_true = df[target].values

    # Carrega o regressor treinado
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo RF de regressão não encontrado em: {MODEL_PATH}")

    regressor = joblib.load(MODEL_PATH)

    # Pede para o modelo chutar a distância para cada amostra
    y_pred = regressor.predict(X)

    # Métricas de erro
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # R² calculado direto
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    print("===== AVALIAÇÃO REGRESSOR RF (DADO SIMULADO) =====")
    print(f"MAE   (erro médio absoluto): {mae:.3f}")
    print(f"RMSE (raiz do erro quadrático médio): {rmse:.3f}")
    print(f"R²    (coeficiente de determinação): {r2:.4f}")
    print("===================================================")

    # Gráfico: distância real vs distância prevista
    max_val = float(max(y_true.max(), y_pred.max()))
    min_val = float(min(y_true.min(), y_pred.min(), 0.0))

    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3, label="amostras")
    plt.plot([min_val, max_val], [min_val, max_val], "--", label="linha ideal (y = x)")

    plt.xlabel("Distância real")
    plt.ylabel("Distância prevista")
    plt.title("Regressor RF (simulação) - real vs previsto")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"Gráfico salvo em: {OUT_FIG}")


if __name__ == "__main__":
    main()
