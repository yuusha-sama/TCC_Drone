import time
import threading
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict

class RFDetectorSim:
    """
    SIMULADOR DE VOO COM ROTEIRO (SCRIPTED FLIGHT).
    Em vez de andar aleatoriamente, o drone segue fases claras:
    1. APROXIMACAO (Longe -> Perto)
    2. PAIRANDO (Fica parado perto -> Perigo)
    3. RETIRADA (Perto -> Longe)
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        
        self.running = False
        self.thread = None
        
        # --- ESTADO DO VOO ---
        self.sim_distance_real = 2000.0 
        self.flight_phase = "APROXIMANDO" # Fases: APROXIMANDO, PAIRANDO, RETIRANDO
        self.hover_counter = 0            # Contador para ficar parado um tempo
        
        # --- ESTADO DO SENSOR ---
        self.latest_rssi = -120
        self.latest_snr = -20
        self.latest_pred_dist = 0.0
        self.latest_prob = 0.0
        self.current_error = 0.0
        self.status = "idle"

        try:
            self.model = joblib.load(self.model_path)
            print(f"RF Sim: Modelo carregado.")
        except Exception as e:
            print(f"RF Sim: Erro no modelo: {e}")
            self.status = "error"

    def start(self):
        if self.model is None: return
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
        print("RF: Iniciando Roteiro de Voo...")
        
        while self.running:
            # 1. LÓGICA DE MOVIMENTO (ROTEIRO)
            velocidade = 30 # m/s (Rápido)
            
            if self.flight_phase == "APROXIMANDO":
                self.sim_distance_real -= velocidade
                # Se chegou perto (50m), muda para PAIRANDO
                if self.sim_distance_real <= 50:
                    self.sim_distance_real = 50
                    self.flight_phase = "PAIRANDO (PERIGO)"
                    self.hover_counter = 20 # Fica 20 ciclos (4 segundos) parado

            elif self.flight_phase == "PAIRANDO (PERIGO)":
                # Não muda a distância, só treme um pouco
                self.sim_distance_real = 50 + np.random.uniform(-2, 2)
                self.hover_counter -= 1
                if self.hover_counter <= 0:
                    self.flight_phase = "RETIRANDO"

            elif self.flight_phase == "RETIRANDO":
                self.sim_distance_real += (velocidade * 1.5) # Foge mais rápido
                if self.sim_distance_real >= 2000:
                    self.flight_phase = "APROXIMANDO" # Reinicia o loop

            # 2. GERAÇÃO DE SINAL (CAMPO ABERTO)
            # RSSI Baseado na distância + Ruído pequeno
            rssi_raw = -20 - (28 * np.log10(self.sim_distance_real + 1)) + np.random.normal(0, 2)
            snr_raw = (rssi_raw + 105) / 3 + np.random.normal(0, 1)
            
            self.latest_rssi = round(rssi_raw, 2)
            self.latest_snr = round(snr_raw, 2)

            # 3. PREDIÇÃO IA
            input_data = pd.DataFrame({'rssi': [self.latest_rssi], 'snr': [self.latest_snr]})
            try:
                pred = self.model.predict(input_data)[0]
                self.latest_pred_dist = float(pred)
            except:
                self.latest_pred_dist = 0.0

            # 4. CÁLCULOS FINAIS
            self.current_error = abs(self.sim_distance_real - self.latest_pred_dist)

            # Probabilidade
            limit_near, limit_far = 100.0, 1000.0
            if self.latest_pred_dist <= limit_near: prob = 1.0
            elif self.latest_pred_dist >= limit_far: prob = 0.0
            else: prob = 1.0 - ((self.latest_pred_dist - limit_near) / (limit_far - limit_near))
            
            self.latest_prob = float(np.clip(prob, 0.0, 1.0))
            
            time.sleep(0.2) # 5 Hz

    def get_latest_metrics(self) -> Dict:
        return {
            "prob": self.latest_prob,
            "dist_m": self.latest_pred_dist,
            "rssi": self.latest_rssi,
            "snr": self.latest_snr,
            "real_dist": self.sim_distance_real,
            "error": self.current_error,
            "status": self.status,
            "phase": self.flight_phase # Manda a fase para a GUI mostrar
        }