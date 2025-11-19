import time
import threading
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict
import serial # Requer: pip install pyserial

class RFDetectorReal:
    """
    DETECTOR RF REAL (HARDWARE VIA USB).
    Conecta na porta Serial (COM3/ttyUSB0) e le dados reais do Dongle.
    """

    def __init__(self, model_path: Path, port: str = "COM3", baudrate: int = 115200):
        self.model_path = model_path
        self.port = port
        self.baudrate = baudrate
        
        self.model = None
        self.serial_conn = None
        
        self.running = False
        self.thread = None
        
        # Estado
        self.latest_rssi = -130
        self.latest_snr = -20
        self.latest_pred_dist = 0.0
        self.latest_prob = 0.0
        self.status = "disconnected"
        self.phase = "AGUARDANDO SINAL" # Para compatibilidade com a GUI

        # Carrega IA
        try:
            self.model = joblib.load(self.model_path)
            print(f"RF Real: Modelo IA carregado.")
        except Exception as e:
            print(f"RF Real: Erro IA: {e}")
            self.status = "error_model"

    def start(self):
        if self.model is None: return
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"RF Real: Conectado na porta {self.port}")
            self.status = "connected"
        except Exception as e:
            print(f"RF Real: Erro ao abrir porta {self.port}: {e}")
            self.status = "connection_error"
            return

        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except:
                pass
        self.status = "disconnected"

    def is_running(self) -> bool:
        return self.running

    def _read_loop(self):
        print("RF Real: Lendo dados da USB...")
        while self.running:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    # Espera receber algo como: "-80,10" (RSSI,SNR)
                    line_bytes = self.serial_conn.readline()
                    line_str = line_bytes.decode('utf-8').strip()
                    
                    if line_str:
                        parts = line_str.split(',')
                        if len(parts) >= 2:
                            rssi = float(parts[0])
                            snr = float(parts[1])
                            
                            self.latest_rssi = rssi
                            self.latest_snr = snr
                            self.phase = "SINAL RECEBIDO"

                            # IA Prediz
                            input_data = pd.DataFrame({'rssi': [rssi], 'snr': [snr]})
                            pred = self.model.predict(input_data)[0]
                            self.latest_pred_dist = float(pred)

                            # Probabilidade
                            limit_near, limit_far = 100.0, 1000.0
                            if self.latest_pred_dist <= limit_near: prob = 1.0
                            elif self.latest_pred_dist >= limit_far: prob = 0.0
                            else: prob = 1.0 - ((self.latest_pred_dist - limit_near) / (limit_far - limit_near))
                            self.latest_prob = float(np.clip(prob, 0.0, 1.0))
                            
                            self.status = "receiving"
                except:
                    self.status = "read_error"
            time.sleep(0.01)

    def get_latest_metrics(self) -> Dict:
        return {
            "prob": self.latest_prob,
            "dist_m": self.latest_pred_dist,
            "rssi": self.latest_rssi,
            "snr": self.latest_snr,
            "real_dist": 0, # Hardware real nao sabe a verdade
            "error": 0,
            "status": self.status,
            "phase": self.phase
        }