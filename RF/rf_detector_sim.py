import time
import threading
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict

class RFDetectorSim:
    """
    SIMULADOR FÍSICO COM VALIDAÇÃO AO VIVO (TCC):
    1. Gera um voo virtual em tempo real (sem fim).
    2. Aplica física (Log-Normal) para gerar RSSI/SNR ruidosos.
    3. Usa a IA para tentar adivinhar a distância baseada no ruído.
    4. COMPARA a posição real com a predita para mostrar o ERRO.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        
        self.running = False
        self.thread = None
        
        # --- ESTADO DA SIMULAÇÃO (A "Verdade") ---
        self.sim_distance_real = 1500.0  # Drone começa longe (1.5km)
        self.approaching = True          # Está se aproximando?
        
        # --- ESTADO DO SENSOR (O que a IA vê) ---
        self.latest_rssi = -120
        self.latest_snr = -20
        self.latest_pred_dist = 0.0
        self.latest_prob = 0.0
        
        # --- VALIDAÇÃO (O que a Banca quer ver) ---
        self.current_error = 0.0         # Diferença entre Real e Predito
        
        self.status = "idle"

        # Carrega o modelo
        try:
            self.model = joblib.load(self.model_path)
            print(f"RF Sim: Modelo IA carregado com sucesso.")
        except Exception as e:
            print(f"RF Sim: ❌ Erro ao carregar modelo: {e}")
            self.status = "error"

    def start(self):
        if self.model is None:
            print("RF Sim: Não posso iniciar sem modelo.")
            return
            
        if not self.running:
            self.running = True
            self.status = "running"
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        self.status = "idle"

    def is_running(self) -> bool:
        return self.running

    def _simulation_loop(self):
        print("RF: Iniciando Loop de Simulação Física...")
        
        while self.running:
            # 1. ATUALIZA A FÍSICA (Move o drone no mundo virtual)
            velocidade = 20 # metros por ciclo (Drone rápido para a demo ser dinâmica)
            
            if self.approaching:
                self.sim_distance_real -= velocidade
                # Se chegou muito perto (20m), começa a se afastar
                if self.sim_distance_real < 20: 
                    self.approaching = False    
            else:
                self.sim_distance_real += velocidade
                # Se foi muito longe (1.8km), volta a se aproximar
                if self.sim_distance_real > 1800: 
                    self.approaching = True       

            # 2. GERA O SINAL (Simula o Chip Semtech LoRa)
            # Fórmula Log-Normal Shadowing (Mesma do treino + Ruído aleatório)
            # Ruído Gaussiano (np.random.normal) simula interferência real
            rssi_raw = -20 - (28 * np.log10(self.sim_distance_real + 1)) + np.random.normal(0, 5)
            snr_raw = (rssi_raw + 105) / 3 + np.random.normal(0, 2)
            
            self.latest_rssi = round(rssi_raw, 2)
            self.latest_snr = round(snr_raw, 2)

            # 3. IA EM AÇÃO (Predição)
            # A IA recebe APENAS o sinal ruidoso, ela NÃO sabe a distância real
            input_data = pd.DataFrame({'rssi': [self.latest_rssi], 'snr': [self.latest_snr]})
            try:
                pred = self.model.predict(input_data)[0]
                self.latest_pred_dist = float(pred)
            except:
                self.latest_pred_dist = 0.0

            # 4. CÁLCULO DO ERRO (Ground Truth vs IA)
            self.current_error = abs(self.sim_distance_real - self.latest_pred_dist)

            # 5. CÁLCULO DE RISCO (Probabilidade para Fusão)
            limit_near, limit_far = 100.0, 1000.0
            
            if self.latest_pred_dist <= limit_near:
                prob = 1.0
            elif self.latest_pred_dist >= limit_far:
                prob = 0.0
            else:
                prob = 1.0 - ((self.latest_pred_dist - limit_near) / (limit_far - limit_near))
            
            self.latest_prob = float(np.clip(prob, 0.0, 1.0))

            # Taxa de atualização (simula leitura do sensor a 5Hz)
            time.sleep(0.2)

    def get_latest_metrics(self) -> Dict:
        """Retorna tudo que a GUI precisa para desenhar os gráficos."""
        return {
            "prob": self.latest_prob,
            "dist_m": self.latest_pred_dist, # O que a IA acha
            "rssi": self.latest_rssi,
            "snr": self.latest_snr,
            "real_dist": self.sim_distance_real, # A verdade (Ground Truth)
            "error": self.current_error,         # A precisão atual
            "status": self.status
        }