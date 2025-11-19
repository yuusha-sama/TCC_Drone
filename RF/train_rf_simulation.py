import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import sys

# --- CONFIGURAÇÃO DE CAMINHOS ---
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def gerar_dados_treino_tcc(n_samples=5000):
    """
    Gera dataset sintético com física LoRa + Ruído
    """
    print("📡 Gerando 5.000 pontos de dados (Log-Normal Shadowing)...")
    distancias = np.random.uniform(5, 2000, n_samples)
    
    # RSSI = Ptx - PathLoss + Shadowing (Ruído)
    # Adicionei um pouco mais de ruído (std=6) para ser realista
    rssi = -20 - (28 * np.log10(distancias + 1)) + np.random.normal(0, 6, n_samples)
    
    # SNR
    snr = (rssi + 105) / 3
    snr = np.clip(snr, -20, 10) + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({'rssi': rssi, 'snr': snr, 'distancia': distancias})
    return df

if __name__ == "__main__":
    # 1. GERAR BASE
    df = gerar_dados_treino_tcc()

    # Salva o CSV para você mostrar na banca
    csv_path = CURRENT_DIR / "dataset_rf_simulado.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Dataset salvo em: {csv_path}")

    # 2. PREPARAR (DIVISÃO TREINO vs TESTE)
    # Isso é crucial para o TCC: Validar em dados que a IA nunca viu
    X = df[['rssi', 'snr']]
    y = df['distancia']

    # Separa 80% para treinar e 20% para provar que funciona
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TREINAR
    print("🧠 Treinando Random Forest (pode levar alguns segundos)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. VALIDAR (CALCULAR MÉTRICAS)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 5. EXIBIR RELATÓRIO (Tire print disso para o TCC)
    print("\n" + "="*40)
    print("📊 RELATÓRIO DE PERFORMANCE DO MODELO RF")
    print("="*40)
    print(f"Erro Médio Absoluto (MAE):  {mae:.2f} metros")
    print(f"--> Significa que o modelo erra a distância por aprox. {mae:.2f}m")
    print("-" * 40)
    print(f"Erro Quadrático Médio (RMSE): {rmse:.2f} metros")
    print("-" * 40)
    print(f"Coeficiente R² (Precisão):  {r2:.4f} (0.0 a 1.0)")
    print(f"--> O modelo explica {r2*100:.1f}% da variância dos dados.")
    print("="*40 + "\n")

    # 6. GERAR GRÁFICO DE VALIDAÇÃO
    print("📈 Gerando gráfico de validação...")
    plt.figure(figsize=(10, 6))
    
    # Plotar Real vs Predito
    plt.scatter(y_test, y_pred, alpha=0.5, s=10, color='blue', label='Predições')
    
    # Linha Ideal (Onde seria o acerto perfeito)
    plt.plot([0, 2000], [0, 2000], 'r--', lw=2, label='Ideal (Perfeito)')
    
    plt.xlabel("Distância Real (Metros)")
    plt.ylabel("Distância Predita pelo Modelo (Metros)")
    plt.title(f"Validação do Modelo RF (R² = {r2:.2f})")
    plt.legend()
    plt.grid(True)
    
    # Salva o gráfico na pasta RF também
    graph_path = CURRENT_DIR / "grafico_performance_rf.png"
    plt.savefig(graph_path)
    print(f"🖼️ Gráfico salvo em: {graph_path}")
    
    # Mostra na tela
    plt.show()

    # 7. SALVAR MODELO FINAL
    output_path = MODELS_DIR / "drone_rf_regressor.pkl"
    joblib.dump(model, output_path)
    print(f"✅ Modelo final salvo em: {output_path}")