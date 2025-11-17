# UI_UX/detector_app_gui.py
#
# Implementar a interface gráfica principal do sistema de detecção:
# - painel à esquerda para o canal RF
# - painel à direita para o canal acústico
# - três botões de controle:
#     * iniciar/parar RF
#     * iniciar/parar acústico
#     * iniciar/parar ambos
# - log de mensagens na parte inferior
#
# Esta camada utiliza:
# - AcousticDetector (acústico), definido em acoustic_detector_core.py
# - RFDetectorStub (RF), definido em rf_detector_stub.py

from pathlib import Path
import threading

import tkinter as tk
from tkinter import ttk, messagebox

from acoustic_detector_core import AcousticDetector
from rf_detector_stub import RFDetectorStub


class DetectorAppGUI:
    """
    Implementar a janela principal do sistema de detecção.

    Layout geral:
    - parte superior: dois painéis lado a lado
        * painel esquerdo: RF
        * painel direito: acústico
    - linha de botões: RF, Acústico, Ambos
    - parte inferior: log de mensagens
    """

    def __init__(self, root: tk.Tk, acoustic_detector: AcousticDetector, rf_detector: RFDetectorStub) -> None:
        self.root = root
        self.acoustic_detector = acoustic_detector
        self.rf_detector = rf_detector

        # Configurar janela principal
        self.root.title("Sistema de Detecção de Drones - Interface Gráfica")
        self.root.geometry("900x500")
        self.root.resizable(False, False)

        # Criar elementos da interface
        self._create_widgets()

        # Iniciar ciclo de atualização periódica
        self._schedule_update()

        # Registrar callback de fechamento da janela
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self) -> None:
        """
        Criar e organizar todos os elementos visuais da interface.
        """
        # Frame principal, com dois painéis lado a lado
        top_frame = ttk.Frame(self.root, padding=(10, 10))
        top_frame.pack(fill="both", expand=True)

        # Painel RF (esquerda)
        rf_frame = ttk.LabelFrame(top_frame, text="Canal RF", padding=(10, 10))
        rf_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.rf_state_label = ttk.Label(
            rf_frame,
            text="RF parado",
            font=("Segoe UI", 12, "bold"),
            foreground="gray",
        )
        self.rf_state_label.pack(anchor="w", pady=(0, 10))

        self.rf_value_label = ttk.Label(
            rf_frame,
            text="Métrica RF: 0.00",
            font=("Segoe UI", 10),
        )
        self.rf_value_label.pack(anchor="w")

        self.rf_status_detail_label = ttk.Label(
            rf_frame,
            text="RF aguardando implementação de hardware.",
            font=("Segoe UI", 9),
            wraplength=400,
        )
        self.rf_status_detail_label.pack(anchor="w", pady=(5, 0))

        # Painel acústico (direita)
        ac_frame = ttk.LabelFrame(top_frame, text="Canal acústico", padding=(10, 10))
        ac_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        self.ac_state_label = ttk.Label(
            ac_frame,
            text="Acústico parado",
            font=("Segoe UI", 12, "bold"),
            foreground="gray",
        )
        self.ac_state_label.pack(anchor="w", pady=(0, 10))

        self.ac_current_prob_label = ttk.Label(
            ac_frame,
            text="Probabilidade atual de drone (acústico): 0.00",
            font=("Segoe UI", 10),
        )
        self.ac_current_prob_label.pack(anchor="w")

        self.ac_avg_prob_label = ttk.Label(
            ac_frame,
            text="Probabilidade média (últimos blocos): 0.00",
            font=("Segoe UI", 10),
        )
        self.ac_avg_prob_label.pack(anchor="w", pady=(2, 0))

        ac_progress_frame = ttk.Frame(ac_frame)
        ac_progress_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(
            ac_progress_frame,
            text="Indicador de probabilidade de drone (acústico):",
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        self.ac_prob_progress = ttk.Progressbar(
            ac_progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
        )
        self.ac_prob_progress.pack(fill="x", pady=(5, 0))

        # Linha de botões (RF, Acústico, Ambos)
        controls_frame = ttk.Frame(self.root, padding=(10, 5))
        controls_frame.pack(fill="x")

        self.btn_rf = ttk.Button(
            controls_frame,
            text="Iniciar RF",
            width=15,
            command=self._toggle_rf,
        )
        self.btn_rf.pack(side="left", padx=(0, 5))

        self.btn_acoustic = ttk.Button(
            controls_frame,
            text="Iniciar acústico",
            width=15,
            command=self._toggle_acoustic,
        )
        self.btn_acoustic.pack(side="left", padx=(5, 5))

        self.btn_both = ttk.Button(
            controls_frame,
            text="Iniciar ambos",
            width=15,
            command=self._toggle_both,
        )
        self.btn_both.pack(side="left", padx=(5, 0))

        # Log de mensagens
        log_frame = ttk.Frame(self.root, padding=(10, 5))
        log_frame.pack(fill="both", expand=True)

        ttk.Label(
            log_frame,
            text="Log de mensagens:",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        self.log_text = tk.Text(
            log_frame,
            height=8,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.log_text.pack(fill="both", expand=True, pady=(5, 0))

    def _log(self, message: str) -> None:
        """
        Adicionar uma linha ao log de mensagens.
        """
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _toggle_rf(self) -> None:
        """
        Ligar ou desligar o canal RF, conforme o estado atual.
        """
        if self.rf_detector.is_running():
            self.rf_detector.stop()
            self._log("Canal RF parado.")
            self.btn_rf.configure(text="Iniciar RF")
        else:
            try:
                self.rf_detector.start()
                self._log("Canal RF iniciado (stub, aguardando integração real).")
                self.btn_rf.configure(text="Parar RF")
            except Exception as exc:
                self._log(f"Erro ao iniciar RF: {exc}")
                messagebox.showerror(
                    "Erro ao iniciar RF",
                    f"Não foi possível iniciar o detector RF.\n\n{exc}",
                )

    def _toggle_acoustic(self) -> None:
        """
        Ligar ou desligar o canal acústico, conforme o estado atual.
        """
        if self.acoustic_detector.is_running():
            self.acoustic_detector.stop()
            self._log("Canal acústico parado.")
            self.btn_acoustic.configure(text="Iniciar acústico")
        else:
            def start_acoustic():
                try:
                    self.acoustic_detector.start()
                    self._log("Canal acústico iniciado.")
                except Exception as exc:
                    self._log(f"Erro ao iniciar acústico: {exc}")
                    messagebox.showerror(
                        "Erro ao iniciar acústico",
                        f"Não foi possível iniciar o detector acústico.\n\n{exc}",
                    )

            thread = threading.Thread(target=start_acoustic, daemon=True)
            thread.start()
            self.btn_acoustic.configure(text="Parar acústico")

    def _toggle_both(self) -> None:
        """
        Ligar ou desligar ambos os canais (RF e acústico).

        Regra:
        - se qualquer um dos dois estiver ligado, a ação é desligar os dois
        - se ambos estiverem desligados, a ação é ligar os dois
        """
        running_any = self.rf_detector.is_running() or self.acoustic_detector.is_running()

        if running_any:
            # Parar ambos
            if self.rf_detector.is_running():
                self.rf_detector.stop()
            if self.acoustic_detector.is_running():
                self.acoustic_detector.stop()
            self._log("Ambos os canais foram parados.")
            self.btn_rf.configure(text="Iniciar RF")
            self.btn_acoustic.configure(text="Iniciar acústico")
            self.btn_both.configure(text="Iniciar ambos")
        else:
            # Iniciar ambos
            self._toggle_rf()
            self._toggle_acoustic()
            self.btn_both.configure(text="Parar ambos")

    def _update_panels(self) -> None:
        """
        Atualizar os painéis RF e acústico com as métricas mais recentes.
        """
        # Atualizar painel RF
        rf_metrics = self.rf_detector.get_latest_metrics()
        rf_status = rf_metrics["status"]
        rf_value = rf_metrics["value"]
        rf_error = rf_metrics["error"]

        self.rf_value_label.configure(
            text=f"Métrica RF: {rf_value:.2f}"
        )

        if rf_status == "running":
            self.rf_state_label.configure(
                text="RF em execução (stub)",
                foreground="darkblue",
            )
        elif rf_status == "error":
            self.rf_state_label.configure(
                text="Erro no RF",
                foreground="darkred",
            )
            if rf_error:
                self._log(f"Erro no RF: {rf_error}")
        else:
            self.rf_state_label.configure(
                text="RF parado",
                foreground="gray",
            )

        # Atualizar painel acústico
        ac_metrics = self.acoustic_detector.get_latest_metrics()
        prob = ac_metrics["prob"]
        avg_prob = ac_metrics["avg_prob"]
        ac_status = ac_metrics["status"]
        ac_error = ac_metrics["error"]

        self.ac_current_prob_label.configure(
            text=f"Probabilidade atual de drone (acústico): {prob:.2f}"
        )
        self.ac_avg_prob_label.configure(
            text=f"Probabilidade média (últimos blocos): {avg_prob:.2f}"
        )
        self.ac_prob_progress["value"] = avg_prob * 100.0

        if ac_status == "running":
            if (
                avg_prob >= self.acoustic_detector.avg_threshold
                or prob >= self.acoustic_detector.strong_threshold
            ):
                self.ac_state_label.configure(
                    text="Drone detectado (acústico)",
                    foreground="darkred",
                )
            else:
                self.ac_state_label.configure(
                    text="Sem drone detectado (acústico)",
                    foreground="darkgreen",
                )
        elif ac_status == "error":
            self.ac_state_label.configure(
                text="Erro no canal acústico",
                foreground="darkred",
            )
            if ac_error:
                self._log(f"Erro no acústico: {ac_error}")
        else:
            self.ac_state_label.configure(
                text="Acústico parado",
                foreground="gray",
            )

    def _schedule_update(self) -> None:
        """
        Agendar atualização periódica da interface.

        A cada 200 ms, atualizar os painéis RF e acústico e reagendar a próxima chamada.
        """
        self._update_panels()
        self.root.after(200, self._schedule_update)

    def _on_close(self) -> None:
        """
        Garantir parada de ambos os detectores ao fechar a janela.
        """
        if self.rf_detector.is_running():
            self.rf_detector.stop()
        if self.acoustic_detector.is_running():
            self.acoustic_detector.stop()
        self.root.destroy()


def main() -> None:
    """
    Função principal da aplicação gráfica.

    Criar instâncias dos detectores RF (stub) e acústico, configurar a janela
    Tkinter e iniciar o loop de eventos.
    """
    model_path = Path("models") / "drone_mfcc_rf.pkl"

    try:
        acoustic_detector = AcousticDetector(model_path=model_path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Erro ao carregar modelo acústico",
            f"Não foi possível carregar o modelo de detecção acústica.\n\n{exc}",
        )
        root.destroy()
        return

    rf_detector = RFDetectorStub()

    root = tk.Tk()
    app = DetectorAppGUI(root, acoustic_detector, rf_detector)
    root.mainloop()


if __name__ == "__main__":
    main()
