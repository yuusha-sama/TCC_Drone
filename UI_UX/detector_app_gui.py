from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import time
import threading
import random

# Ajuste de Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sensor_fusion_core import SensorFusionEngine

# Importa os módulos de sensores
try:
    from acoustic_detector_core import AcousticDetector
except ImportError:
    AcousticDetector = None

try:
    from RF.rf_detector_sim import RFDetectorSim
    from RF.rf_detector_real import RFDetectorReal
except ImportError as e:
    sys.exit(f"ERRO CRITICO: Modulos RF nao encontrados. {e}")


# --- MOCK DE ÁUDIO (Para o botão 'Simulação Acústica') ---
class MockAcousticDetector:
    def __init__(self):
        self.running = False
        self.latest_prob = 0.0
        self.avg_prob = 0.0
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running

    def _loop(self):
        while self.running:
            # Simula picos de deteccao aleatórios
            if random.random() > 0.85:
                self.latest_prob = random.uniform(0.3, 0.95)
            else:
                self.latest_prob = random.uniform(0.0, 0.15)
            
            self.avg_prob = (self.avg_prob * 0.7) + (self.latest_prob * 0.3)
            time.sleep(0.5)

    def get_latest_metrics(self):
        return {
            "prob": self.latest_prob,
            "avg_prob": self.avg_prob,
            "status": "simulating"
        }


class DetectorAppGUI:
    def __init__(self, root, rf_model_path, ac_model_path, rf_port="COM5"):
        self.root = root
        self.root.title("Sistema Anti-Drone TCC (Multimodo)")
        self.root.geometry("1100x800")

        # Configurações
        self.rf_model_path = rf_model_path
        self.ac_model_path = ac_model_path
        self.rf_port = rf_port

        # Objetos dos sensores (começam vazios)
        self.rf_detector = None
        self.acoustic_detector = None
        self.fusion_engine = SensorFusionEngine()

        # Estilos
        self._setup_styles()
        
        # Layout
        self._create_layout()
        
        # Loop de atualização
        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.log_message("Sistema pronto. Selecione o modo de operação para cada sensor.")

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        self.style.configure("Danger.TLabel", foreground="red", font=("Arial", 14, "bold"))
        self.style.configure("Safe.TLabel", foreground="green", font=("Arial", 14, "bold"))
        self.style.configure("Good.TLabel", foreground="green", font=("Arial", 10))
        self.style.configure("Bad.TLabel", foreground="red", font=("Arial", 10))
        self.style.configure("AcDetected.TLabel", foreground="red", font=("Arial", 11, "bold"))
        self.style.configure("AcListening.TLabel", foreground="blue", font=("Arial", 10))
        
        # Botões coloridos (opcional, depende do tema)
        self.style.configure("Sim.TButton", foreground="blue")
        self.style.configure("Real.TButton", foreground="darkred")

    def _create_layout(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Painel de Controle Multimodal", style="Header.TLabel").pack(pady=5)

        # === ÁREA DOS SENSORES ===
        sensors_frame = ttk.Frame(main_frame)
        sensors_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # PAINEL RF (ESQUERDA)
        self.rf_frame = ttk.LabelFrame(sensors_frame, text="Sensor RF (LRS)", padding="10")
        self.rf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Status
        self.lbl_rf_mode = ttk.Label(self.rf_frame, text="Modo: DESLIGADO", foreground="gray")
        self.lbl_rf_mode.pack(anchor="w")
        self.lbl_rf_phase = ttk.Label(self.rf_frame, text="Status: --", foreground="blue")
        self.lbl_rf_phase.pack(anchor="w")
        
        ttk.Separator(self.rf_frame, orient='horizontal').pack(fill='x', pady=5)

        # Dados
        self.lbl_rf_rssi = ttk.Label(self.rf_frame, text="RSSI: -- dBm")
        self.lbl_rf_rssi.pack(anchor="w")
        self.lbl_rf_real = ttk.Label(self.rf_frame, text="Dist. Real: -- m")
        self.lbl_rf_real.pack(anchor="w")
        self.lbl_rf_pred = ttk.Label(self.rf_frame, text="Dist. IA: -- m", font=("Arial", 10, "bold"))
        self.lbl_rf_pred.pack(anchor="w")
        self.lbl_rf_error = ttk.Label(self.rf_frame, text="Erro: -- m", style="Good.TLabel")
        self.lbl_rf_error.pack(anchor="w")
        
        # Risco
        self.lbl_rf_prob = ttk.Label(self.rf_frame, text="Risco RF: 0.0%")
        self.lbl_rf_prob.pack(anchor="w", pady=5)
        self.progress_rf = ttk.Progressbar(self.rf_frame, length=100, mode="determinate")
        self.progress_rf.pack(fill=tk.X, pady=5)
        
        # BOTÕES RF
        btn_rf_frame = ttk.Frame(self.rf_frame)
        btn_rf_frame.pack(fill=tk.X, pady=10)
        
        self.btn_rf_sim = ttk.Button(btn_rf_frame, text="Simulação", style="Sim.TButton", command=self._start_rf_sim)
        self.btn_rf_sim.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_rf_real = ttk.Button(btn_rf_frame, text="Hardware USB", style="Real.TButton", command=self._start_rf_real)
        self.btn_rf_real.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_rf_stop = ttk.Button(self.rf_frame, text="PARAR RF", command=self._stop_rf)
        self.btn_rf_stop.pack(fill=tk.X)

        # PAINEL ACÚSTICO (DIREITA)
        self.ac_frame = ttk.LabelFrame(sensors_frame, text="Sensor Acústico", padding="10")
        self.ac_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Status
        self.lbl_ac_mode = ttk.Label(self.ac_frame, text="Modo: DESLIGADO", foreground="gray")
        self.lbl_ac_mode.pack(anchor="w")
        self.lbl_ac_status = ttk.Label(self.ac_frame, text="Status: --")
        self.lbl_ac_status.pack(anchor="w")
        
        ttk.Separator(self.ac_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Dados
        self.lbl_ac_prob = ttk.Label(self.ac_frame, text="Prob. Inst.: 0.0%")
        self.lbl_ac_prob.pack(anchor="w")
        self.lbl_ac_avg = ttk.Label(self.ac_frame, text="Média (5s): 0.0%")
        self.lbl_ac_avg.pack(anchor="w", pady=5)
        
        self.progress_ac = ttk.Progressbar(self.ac_frame, length=100, mode="determinate")
        self.progress_ac.pack(fill=tk.X, pady=5)
        
        # BOTÕES ACÚSTICO
        btn_ac_frame = ttk.Frame(self.ac_frame)
        btn_ac_frame.pack(fill=tk.X, pady=10)
        
        self.btn_ac_sim = ttk.Button(btn_ac_frame, text="Simulação", style="Sim.TButton", command=self._start_ac_sim)
        self.btn_ac_sim.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_ac_real = ttk.Button(btn_ac_frame, text="Microfone", style="Real.TButton", command=self._start_ac_real)
        self.btn_ac_real.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_ac_stop = ttk.Button(self.ac_frame, text="PARAR ÁUDIO", command=self._stop_ac)
        self.btn_ac_stop.pack(fill=tk.X)

        # FUSÃO (CENTRO)
        fusion_frame = ttk.LabelFrame(main_frame, text="Decisão de Fusão", padding="10")
        fusion_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_fusion_prob = ttk.Label(fusion_frame, text="Probabilidade Final: 0.00%")
        self.lbl_fusion_prob.pack()
        self.progress_fusion = ttk.Progressbar(fusion_frame, length=100, mode="determinate")
        self.progress_fusion.pack(fill=tk.X, pady=5)
        self.lbl_fusion_status = ttk.Label(fusion_frame, text="SISTEMA AGUARDANDO", style="Header.TLabel", foreground="gray")
        self.lbl_fusion_status.pack()

        # LOG 
        log_frame = ttk.LabelFrame(main_frame, text="Log do Sistema", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.txt_log = tk.Text(log_frame, height=8, state="disabled", font=("Consolas", 9))
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.txt_log, command=self.txt_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log['yscrollcommand'] = scrollbar.set

    # === LÓGICA DE CONTROLE RF ===
    def _stop_rf(self):
        if self.rf_detector:
            self.rf_detector.stop()
            self.rf_detector = None
        self.lbl_rf_mode.config(text="Modo: DESLIGADO", foreground="gray")
        self.lbl_rf_phase.config(text="Status: --", foreground="black")
        self.log_message("Sensor RF parado.")

    def _start_rf_sim(self):
        self._stop_rf() # Garante que parou o anterior
        self.log_message("Iniciando RF em modo SIMULAÇÃO...")
        self.rf_detector = RFDetectorSim(model_path=self.rf_model_path)
        self.rf_detector.start()
        self.lbl_rf_mode.config(text="Modo: SIMULAÇÃO (Física)", foreground="blue")

    def _start_rf_real(self):
        self._stop_rf()
        self.log_message(f"Conectando RF HARDWARE na porta {self.rf_port}...")
        try:
            self.rf_detector = RFDetectorReal(model_path=self.rf_model_path, port=self.rf_port)
            self.rf_detector.start()
            self.lbl_rf_mode.config(text=f"Modo: REAL ({self.rf_port})", foreground="darkred")
        except Exception as e:
            self.log_message(f"Erro ao conectar Hardware: {e}")
            self._stop_rf()

    # === LÓGICA DE CONTROLE ACÚSTICO ===
    def _stop_ac(self):
        if self.acoustic_detector:
            self.acoustic_detector.stop()
            self.acoustic_detector = None
        self.lbl_ac_mode.config(text="Modo: DESLIGADO", foreground="gray")
        self.lbl_ac_status.config(text="Status: --", style="TLabel")
        self.log_message("Sensor Acústico parado.")

    def _start_ac_sim(self):
        self._stop_ac()
        self.log_message("Iniciando Áudio em modo SIMULAÇÃO (Mock)...")
        self.acoustic_detector = MockAcousticDetector()
        self.acoustic_detector.start()
        self.lbl_ac_mode.config(text="Modo: SIMULAÇÃO (Mock)", foreground="blue")

    def _start_ac_real(self):
        self._stop_ac()
        if AcousticDetector is None:
            self.log_message("Erro: Bibliotecas de áudio não instaladas/importadas.")
            return
            
        self.log_message("Ligando MICROFONE REAL...")
        try:
            self.acoustic_detector = AcousticDetector(model_path=self.ac_model_path)
            self.acoustic_detector.start()
            self.lbl_ac_mode.config(text="Modo: MICROFONE REAL", foreground="darkred")
        except Exception as e:
            self.log_message(f"Erro ao abrir Mic: {e}")
            self._stop_ac()

    # === ATUALIZAÇÃO DA GUI ===
    def log_message(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    def _schedule_update(self):
        # 1. ATUALIZAÇÃO RF
        p_rf = 0.0
        rf_active = False
        
        if self.rf_detector and self.rf_detector.is_running():
            rf_active = True
            rf = self.rf_detector.get_latest_metrics()
            
            # Atualiza textos
            phase = rf.get('phase', '--')
            self.lbl_rf_phase.config(text=f"Status: {phase}")
            
            self.lbl_rf_rssi.config(text=f"RSSI: {rf.get('rssi',0):.1f} dBm")
            self.lbl_rf_real.config(text=f"Dist. Real: {rf.get('real_dist',0):.0f} m")
            self.lbl_rf_pred.config(text=f"Dist. IA: {rf.get('dist_m',0):.0f} m")
            
            # Erro
            erro = rf.get('error', 0)
            style_err = "Bad.TLabel" if erro > 100 else "Good.TLabel"
            self.lbl_rf_error.config(text=f"Erro: {erro:.1f} m", style=style_err)
            
            p_rf = rf.get('prob', 0)
            self.lbl_rf_prob.config(text=f"Risco RF: {p_rf*100:.0f}%")
            self.progress_rf["value"] = p_rf * 100

        # 2. ATUALIZAÇÃO ACÚSTICA
        p_ac_avg = 0.0
        ac_active = False
        
        if self.acoustic_detector and self.acoustic_detector.is_running():
            ac_active = True
            ac = self.acoustic_detector.get_latest_metrics()
            p_ac_avg = ac.get('avg_prob', 0)
            
            self.lbl_ac_prob.config(text=f"Prob. Inst.: {ac.get('prob',0)*100:.0f}%")
            self.lbl_ac_avg.config(text=f"Média: {p_ac_avg*100:.1f}%")
            self.progress_ac["value"] = p_ac_avg * 100
            
            # Detecção > 20%
            if p_ac_avg >= 0.20:
                self.lbl_ac_status.config(text="Status: DRONE DETECTADO!", style="AcDetected.TLabel")
            else:
                self.lbl_ac_status.config(text="Status: Ouvindo...", style="AcListening.TLabel")

        # 3. FUSÃO
        self.fusion_engine.update(
            acoustic_prob=p_ac_avg,
            rf_prob=p_rf,
            acoustic_active=ac_active,
            rf_active=rf_active
        )
        
        fus = self.fusion_engine.get_latest_metrics()
        p_final = fus.get('fused_prob', 0)
        
        self.lbl_fusion_prob.config(text=f"Probabilidade Final: {p_final*100:.1f}%")
        self.progress_fusion["value"] = p_final * 100
        
        # Lógica visual de status final
        if not rf_active and not ac_active:
            self.lbl_fusion_status.config(text="AGUARDANDO SENSORES", style="Header.TLabel", foreground="gray")
        elif p_final >= 0.6:
            self.lbl_fusion_status.config(text=" DRONE DETECTADO", style="Danger.TLabel")
        elif p_final > 0.2:
            self.lbl_fusion_status.config(text=" POSSÍVEL AMEAÇA", style="Header.TLabel", foreground="orange")
        else:
            self.lbl_fusion_status.config(text=" ÁREA SEGURA", style="Safe.TLabel")

        # Log de Alerta (evita spam)
        if p_final >= 0.6:
            # Lógica simples para não floodar o log a cada 100ms
            if int(time.time()) % 2 == 0: # Loga a cada ~2 segundos se manter o alerta
                 # Opcional: self.log_message(f"ALERTA CRÍTICO: Probabilidade {p_final*100:.0f}%")
                 pass

        self.root.after(100, self._schedule_update)

    def _on_close(self):
        self._stop_rf()
        self._stop_ac()
        self.root.destroy()

def main():
    # CAMINHOS DOS MODELOS
    rf_model = Path("models") / "drone_rf_regressor.pkl"
    ac_model = Path("models") / "drone_mfcc_rf.pkl"
    
    # CONFIGURAÇÃO PADRÃO DO HARDWARE
    PORTA_USB_PADRAO = "COM5" # Ajuste conforme seu PC

    if not rf_model.exists():
        messagebox.showerror("Erro", "Modelo RF não encontrado. Rode 'train_rf_simulation.py'.")
        return

    root = tk.Tk()
    # Passamos as configurações para a GUI, ela decide o que iniciar
    app = DetectorAppGUI(root, rf_model, ac_model, rf_port=PORTA_USB_PADRAO)
    root.mainloop()

if __name__ == "__main__":
    main()