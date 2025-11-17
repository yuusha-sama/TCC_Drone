# UI_UX/acoustic_detector_gui.py
#
# Implementar a interface gráfica (Tkinter) para o detector acústico.
# Esta camada consome a classe AcousticDetector e exibe:
# - estado atual (drone detectado / sem drone / erro)
# - probabilidade atual e média
# - barra de progresso
# - log de mensagens
# - botões de iniciar/parar

import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

from acoustic_detector_core import AcousticDetector


class AcousticDetectorGUI:
    """
    Implementar a interface gráfica baseada em Tkinter.

    Responsabilidades:
    - criar e organizar os componentes visuais
    - iniciar/parar o detector em threads separadas
    - consultar periodicamente as métricas do detector
    - atualizar os elementos visuais de acordo com o estado do sistema
    """

    def __init__(self, root: tk.Tk, detector: AcousticDetector) -> None:
        self.root = root
        self.detector = detector

        # Configurar janela principal
        self.root.title("Sistema de Detecção Acústica de Drones")
        self.root.geometry("640x420")
        self.root.resizable(False, False)

        # Inicializar elementos da interface
        self._create_widgets()

        # Iniciar ciclo de atualização periódica
        self._schedule_update()

        # Registrar callback para fechamento da janela
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self) -> None:
        """
        Criar e posicionar os elementos da interface gráfica.
        """
        # Cabeçalho
        header_frame = ttk.Frame(self.root, padding=(10, 10))
        header_frame.pack(fill="x")

        title_label = ttk.Label(
            header_frame,
            text="Detecção Acústica de Drones",
            font=("Segoe UI", 16, "bold"),
        )
        title_label.pack(anchor="center")

        subtitle_label = ttk.Label(
            header_frame,
            text=(
                "Monitorar em tempo real a probabilidade de presença de drones "
                "a partir do sinal acústico ambiente."
            ),
            font=("Segoe UI", 9),
            wraplength=600,
        )
        subtitle_label.pack(anchor="center", pady=(5, 0))

        # Área de estado e probabilidades
        status_frame = ttk.Frame(self.root, padding=(10, 10))
        status_frame.pack(fill="x")

        self.state_label = ttk.Label(
            status_frame,
            text="Sistema parado",
            font=("Segoe UI", 14, "bold"),
            foreground="gray",
        )
        self.state_label.pack(anchor="center", pady=(0, 10))

        probs_frame = ttk.Frame(status_frame)
        probs_frame.pack(fill="x", pady=(0, 10))

        self.current_prob_label = ttk.Label(
            probs_frame,
            text="Probabilidade atual de drone: 0.00",
            font=("Segoe UI", 11),
        )
        self.current_prob_label.pack(anchor="w")

        self.avg_prob_label = ttk.Label(
            probs_frame,
            text="Probabilidade média (últimos blocos): 0.00",
            font=("Segoe UI", 11),
        )
        self.avg_prob_label.pack(anchor="w", pady=(2, 0))

        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(fill="x", pady=(10, 10))

        ttk.Label(
            progress_frame,
            text="Indicador de probabilidade de drone:",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        self.prob_progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
        )
        self.prob_progress.pack(fill="x", pady=(5, 0))

        # Botões de controle
        controls_frame = ttk.Frame(self.root, padding=(10, 10))
        controls_frame.pack(fill="x")

        self.start_button = ttk.Button(
            controls_frame,
            text="Iniciar detecção",
            command=self._on_start,
            width=18,
        )
        self.start_button.pack(side="left", padx=(0, 5))

        self.stop_button = ttk.Button(
            controls_frame,
            text="Parar detecção",
            command=self._on_stop,
            width=18,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(5, 0))

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
        Adicionar linha ao log de mensagens na parte inferior da interface.
        """
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_start(self) -> None:
        """
        Callback do botão "Iniciar detecção".

        Iniciar o detector em uma thread separada para não bloquear o loop
        de eventos do Tkinter.
        """
        if self.detector.is_running():
            return

        def start_detector():
            try:
                self.detector.start()
                self._log("Detecção iniciada.")
            except Exception as exc:
                self._log(f"Erro ao iniciar detecção: {exc}")
                messagebox.showerror(
                    "Erro ao iniciar",
                    f"Não foi possível iniciar a detecção.\n\n{exc}",
                )

        thread = threading.Thread(target=start_detector, daemon=True)
        thread.start()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

    def _on_stop(self) -> None:
        """
        Callback do botão "Parar detecção".
        """
        if not self.detector.is_running():
            return

        self.detector.stop()
        self._log("Detecção parada.")

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _update_status_labels(self, metrics: dict) -> None:
        """
        Atualizar textos, cores e barra de progresso com base nas
        métricas recentes fornecidas pelo detector.
        """
        prob = metrics["prob"]
        avg_prob = metrics["avg_prob"]
        status = metrics["status"]
        error_msg = metrics["error"]

        # Atualizar textos de probabilidade
        self.current_prob_label.configure(
            text=f"Probabilidade atual de drone: {prob:.2f}"
        )
        self.avg_prob_label.configure(
            text=f"Probabilidade média (últimos blocos): {avg_prob:.2f}"
        )

        # Converter probabilidade média em porcentagem na barra
        self.prob_progress["value"] = avg_prob * 100.0

        # Atualizar rótulo de estado principal
        if status == "error":
            self.state_label.configure(
                text="Erro na detecção",
                foreground="darkred",
            )
            if error_msg:
                self._log(f"Erro no callback de áudio: {error_msg}")
        elif status == "running":
            if (
                avg_prob >= self.detector.avg_threshold
                or prob >= self.detector.strong_threshold
            ):
                # Situação em que o sistema considera presença de drone
                self.state_label.configure(
                    text="Drone detectado",
                    foreground="darkred",
                )
            else:
                self.state_label.configure(
                    text="Sem drone detectado",
                    foreground="darkgreen",
                )
        else:
            self.state_label.configure(
                text="Sistema parado",
                foreground="gray",
            )

    def _schedule_update(self) -> None:
        """
        Agendar atualização periódica da interface.

        A cada 200 ms, consultar as métricas do detector e atualizar
        os elementos visuais (estado, textos e barra).
        """
        if self.detector is not None:
            metrics = self.detector.get_latest_metrics()
            self._update_status_labels(metrics)

        # Reagendar chamada
        self.root.after(200, self._schedule_update)

    def _on_close(self) -> None:
        """
        Callback executado ao solicitar o fechamento da janela.

        Garantir parada do detector e destruição da janela.
        """
        if self.detector is not None and self.detector.is_running():
            self.detector.stop()
        self.root.destroy()


def main() -> None:
    """
    Função principal da aplicação gráfica.

    Criar instância de AcousticDetector, configurar a janela Tkinter
    e iniciar o loop de eventos.
    """
    model_path = Path("models") / "drone_mfcc_rf.pkl"

    try:
        detector = AcousticDetector(model_path=model_path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Erro ao carregar modelo",
            f"Não foi possível carregar o modelo de detecção.\n\n{exc}",
        )
        root.destroy()
        return

    root = tk.Tk()
    gui = AcousticDetectorGUI(root, detector)
    root.mainloop()


if __name__ == "__main__":
    main()
