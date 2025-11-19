# UI_UX/detector_app_gui.py
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# --- AJUSTE DE PATH ---
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Imports do Projeto
from acoustic_detector_core import AcousticDetector
from sensor_fusion_core import SensorFusionEngine

# Importa o Simulador da pasta RF
try:
    from RF.rf_detector_sim import RFDetectorSim
except ImportError as e:
    print(f"ERRO CRÍTICO: Não achei RF/rf_detector_sim.py. {e}")
    sys.exit(1)


class DetectorAppGUI:
    """
    Interface Gráfica Principal - TCC Drone Detection
    Exibe: Dados RF (Real vs Predito), Acústico e Fusão.
    """
    def __init__(self, root, rf_detector, acoustic_detector, fusion_engine):
        self.root = root
        self.root.title("Sistema de Detecção e Validação (TCC)")
        self.root.geometry("950x700")

        self.rf_detector = rf_detector
        self.acoustic_detector = acoustic_detector
        self.fusion_engine = fusion_engine

        # --- ESTILOS VISUAIS ---
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Fontes e Cores
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        
        # Estilos dinâmicos para o erro
        self.style.configure("Good.TLabel", foreground="green", font=("Arial", 10, "bold"))
        self.style.configure("Bad.TLabel", foreground="red", font=("Arial", 10, "bold"))
        
        # Estilos para Alertas
        self.style.configure("Safe.TLabel", foreground="green", font=("Arial", 16, "bold"))
        self.style.configure("Danger.TLabel", foreground="red", font=("Arial", 16, "bold"))

        self._create_layout()
        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_layout(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        ttk.Label(main_frame, text="Monitoramento de Sensores & Validação IA", style="Header.TLabel").pack(pady=10)

        # --- ÁREA DOS SENSORES ---
        sensors_frame = ttk.Frame(main_frame)
        sensors_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # === PAINEL ESQUERDO: RF (LRS) ===
        self.rf_frame = ttk.LabelFrame(sensors_frame, text="📡 Sensor RF (Simulação Física)", padding="10")
        self.rf_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Dados Brutos
        self.lbl_rf_rssi = ttk.Label(self.rf_frame, text="RSSI: -- dBm")
        self.lbl_rf_rssi.pack(anchor="w")
        self.lbl_rf_snr = ttk.Label(self.rf_frame, text="SNR: -- dB")
        self.lbl_rf_snr.pack(anchor="w")
        
        ttk.Separator(self.rf_frame, orient='horizontal').pack(fill='x', pady=8)
        
        # --- ÁREA DE VALIDAÇÃO (Onde você ganha nota no TCC) ---
        ttk.Label(self.rf_frame, text="[VALIDAÇÃO EM TEMPO REAL]", font=("Arial", 8, "bold"), foreground="gray").pack(anchor="w")
        
        self.lbl_rf_real = ttk.Label(self.rf_frame, text="Distância REAL: -- m")
        self.lbl_rf_real.pack(anchor="w")
        
        self.lbl_rf_pred = ttk.Label(self.rf_frame, text="Distância IA: -- m", font=("Arial", 11, "bold"))
        self.lbl_rf_pred.pack(anchor="w")
        
        self.lbl_rf_error = ttk.Label(self.rf_frame, text="Erro Absoluto: -- m", style="Good.TLabel")
        self.lbl_rf_error.pack(anchor="w", pady=2)
        # -------------------------------------------------------

        ttk.Separator(self.rf_frame, orient='horizontal').pack(fill='x', pady=8)

        self.lbl_rf_prob = ttk.Label(self.rf_frame, text="Nível de Ameaça: 0.0%")
        self.lbl_rf_prob.pack(anchor="w")
        self.progress_rf = ttk.Progressbar(self.rf_frame, length=100, mode="determinate")
        self.progress_rf.pack(fill=tk.X, pady=5)

        # === PAINEL DIREITO: ACÚSTICO ===
        self.ac_frame = ttk.LabelFrame(sensors_frame, text="🎤 Sensor Acústico (MFCC)", padding="10")
        self.ac_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.lbl_ac_prob = ttk.Label(self.ac_frame, text="Prob Instantânea: 0.0%")
        self.lbl_ac_prob.pack(anchor="w")
        
        self.lbl_ac_avg = ttk.Label(self.ac_frame, text="Média Móvel (5s): 0.0%")
        self.lbl_ac_avg.pack(anchor="w")
        
        self.progress_ac = ttk.Progressbar(self.ac_frame, length=100, mode="determinate")
        self.progress_ac.pack(fill=tk.X, pady=5)

        # === PAINEL INFERIOR: FUSÃO ===
        fusion_frame = ttk.LabelFrame(main_frame, text="🧠 Fusão de Sensores (Decisão Final)", padding="10")
        fusion_frame.pack(fill=tk.X, pady=10)
        
        self.lbl_fusion_status = ttk.Label(fusion_frame, text="SISTEMA AGUARDANDO", style="Header.TLabel")
        self.lbl_fusion_status.pack()
        
        self.lbl_fusion_prob = ttk.Label(fusion_frame, text="Probabilidade Combinada: 0.00%")
        self.lbl_fusion_prob.pack()
        
        self.progress_fusion = ttk.Progressbar(fusion_frame, length=100, mode="determinate")
        self.progress_fusion.pack(fill=tk.X, pady=5)

        # Botão de Início
        ttk.Button(main_frame, text="▶ INICIAR SISTEMA COMPLETO", command=self._start_all).pack(fill=tk.X, pady=5, padx=50)

    def _start_all(self):
        """Inicia as threads dos sensores"""
        print("GUI: Iniciando sensores...")
        self.rf_detector.start()
        if self.acoustic_detector:
            try:
                self.acoustic_detector.start()
            except Exception as e:
                print(f"GUI: Erro ao iniciar mic: {e}")

    def _schedule_update(self):
        """Loop principal da GUI (Atualiza a cada 100ms)"""
        
        # 1. Atualiza RF
        rf = self.rf_detector.get_latest_metrics()
        
        # Atualiza textos
        self.lbl_rf_rssi.config(text=f"RSSI: {rf.get('rssi',0):.1f} dBm")
        self.lbl_rf_snr.config(text=f"SNR: {rf.get('snr',0):.1f} dB")
        
        # Lógica de Validação (Real vs IA)
        real = rf.get('real_dist', 0)
        pred = rf.get('dist_m', 0)
        erro = rf.get('error', 0)
        
        self.lbl_rf_real.config(text=f"Distância REAL: {real:.1f} m")
        self.lbl_rf_pred.config(text=f"Distância IA: {pred:.1f} m")
        
        # Se o erro for grande (>50m), fica vermelho. Se pequeno, verde.
        style_err = "Bad.TLabel" if erro > 50 else "Good.TLabel"
        self.lbl_rf_error.config(text=f"Erro (Acurácia): {erro:.1f} m", style=style_err)
        
        # Probabilidade RF
        p_rf = rf.get('prob', 0)
        self.lbl_rf_prob.config(text=f"Nível de Ameaça: {p_rf*100:.1f}%")
        self.progress_rf["value"] = p_rf * 100

        # 2. Atualiza Acústico
        p_ac_avg = 0.0
        if self.acoustic_detector:
            ac = self.acoustic_detector.get_latest_metrics()
            p_ac_inst = ac.get('prob', 0)
            p_ac_avg = ac.get('avg_prob', 0)
            
            self.lbl_ac_prob.config(text=f"Prob Inst: {p_ac_inst*100:.0f}%")
            self.lbl_ac_avg.config(text=f"Média Móvel: {p_ac_avg*100:.1f}%")
            self.progress_ac["value"] = p_ac_avg * 100

        # 3. Executa Fusão
        self.fusion_engine.update(acoustic_prob=p_ac_avg, rf_prob=p_rf)
        fus = self.fusion_engine.get_latest_metrics()
        p_final = fus.get('fused_prob', 0)
        
        self.lbl_fusion_prob.config(text=f"Probabilidade Combinada: {p_final*100:.1f}%")
        self.progress_fusion["value"] = p_final * 100
        
        # Atualiza Status Visual
        if p_final >= 0.6:
            self.lbl_fusion_status.config(text="⚠️ ALERTA: DRONE DETECTADO ⚠️", style="Danger.TLabel")
        elif p_final > 0.3:
             self.lbl_fusion_status.config(text="👁️ ATENÇÃO: POSSÍVEL ATIVIDADE", style="Header.TLabel")
        else:
            self.lbl_fusion_status.config(text="✅ ÁREA SEGURA", style="Safe.TLabel")

        # Re-agenda para daqui 100ms
        self.root.after(100, self._schedule_update)

    def _on_close(self):
        self.rf_detector.stop()
        if self.acoustic_detector:
            self.acoustic_detector.stop()
        self.root.destroy()


def main():
    # 1. Caminho dos Modelos
    rf_model_path = Path("models") / "drone_rf_regressor.pkl"
    ac_model_path = Path("models") / "drone_mfcc_rf.pkl"

    # 2. Inicia o Simulador RF
    if not rf_model_path.exists():
        messagebox.showerror("Erro", "Modelo RF não encontrado! Rode o treino primeiro.")
        return
        
    print(f"Iniciando Simulador RF com modelo: {rf_model_path}")
    rf_detector = RFDetectorSim(model_path=rf_model_path)

    # 3. Inicia o Detector Acústico
    try:
        acoustic_detector = AcousticDetector(model_path=ac_model_path)
        print("Detector Acústico carregado.")
    except Exception as e:
        print(f"AVISO: Detector Acústico falhou ({e}). O app rodará sem mic.")
        acoustic_detector = None

    # 4. Motor de Fusão
    fusion_engine = SensorFusionEngine(weight_acoustic=0.5, weight_rf=0.5)

    # 5. Inicia App
    root = tk.Tk()
    app = DetectorAppGUI(root, rf_detector, acoustic_detector, fusion_engine)
    root.mainloop()

if __name__ == "__main__":
    main()