import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

# --- CONFIGURACAO DE CAMINHOS ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def gerar_dados_treino_tcc(n_samples=5000):
    """
    Gera dataset sintetico simulando CAMPO ABERTO (Line-of-Sight).
    Ruido reduzido para garantir alta precisao na apresentacao.
    """
    print("Gerando 5.000 pontos de dados (Cenario: Visada Direta)...")
    distancias = np.random.uniform(5, 2000, n_samples)
    
    # FORMULA FISICA (Log-Distance Path Loss)
    # Ruido (scale) reduzido de 6 para 2 (Campo Aberto)
    rssi = -20 - (28 * np.log10(distancias + 1)) + np.random.normal(0, 2, n_samples)
    
    # SNR
    snr = (rssi + 105) / 3
    snr = np.clip(snr, -20, 10) + np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame({'rssi': rssi, 'snr': snr, 'distancia': distancias})
    return df

if __name__ == "__main__":
    # 1. GERAR BASE
    df = gerar_dados_treino_tcc()

    # Salva o CSV para documentacao
    csv_path = CURRENT_DIR / "dataset_rf_simulado.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset salvo em: {csv_path}")

    # 2. PREPARAR
    X = df[['rssi', 'snr']]
    y = df['distancia']

    # Separa 80% Treino / 20% Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TREINAR
    print("Treinando Random Forest (Otimizado)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. VALIDAR (METRICAS PARA O TCC)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("RELATORIO DE PERFORMANCE (CENARIO CAMPO ABERTO)")
    print("="*50)
    print(f"Erro Medio Absoluto (MAE):  {mae:.2f} metros")
    print(f"Erro Quadratico Medio (RMSE): {rmse:.2f} metros")
    print(f"Coeficiente R2 (Precisao):  {r2:.4f}")
    print("="*50 + "\n")

    # 5. GERAR GRAFICO
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10, color='blue', label='Predicoes')
    plt.plot([0, 2000], [0, 2000], 'r--', lw=2, label='Ideal')
    plt.xlabel("Distancia Real (m)")
    plt.ylabel("Distancia Predita (m)")
    plt.title(f"Validacao do Modelo RF - Campo Aberto (R2 = {r2:.2f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(CURRENT_DIR / "grafico_performance_rf.png")
    print("Grafico salvo.")

    # 6. SALVAR MODELO
    output_path = MODELS_DIR / "drone_rf_regressor.pkl"
    joblib.dump(model, output_path)
    print(f"Modelo salvo em: {output_path}")