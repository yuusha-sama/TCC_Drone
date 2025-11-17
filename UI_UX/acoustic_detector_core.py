# UI_UX/acoustic_detector_core.py
#
# Implementar a lógica de detecção acústica:
# - captura de áudio com sounddevice
# - extração de MFCCs
# - classificação com modelo salvo (RandomForest)
# Esta camada não conhece interface gráfica.

from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
import sounddevice as sd
import joblib
import sys

# Incluir diretório raiz do projeto no sys.path para permitir import DB.*
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DB.audio_mfcc_utils import extract_mfcc_from_buffer  # noqa: E402


class AcousticDetector:
    """
    Encapsular a lógica de captura de áudio e classificação.

    Responsabilidades:
    - gerenciar o stream de entrada de áudio (start/stop)
    - extrair MFCC de cada bloco recebido
    - chamar o modelo treinado e calcular probabilidade de drone
    - armazenar métricas recentes para consumo externo (GUI ou CLI)
    """

    def __init__(
        self,
        model_path: Path,
        sample_rate: int = 16000,
        block_duration: float = 1.0,
        history_size: int = 5,
        avg_threshold: float = 0.30,
        strong_threshold: float = 0.60,
        drone_label: str = "Drone",
    ) -> None:
        # Parâmetros de modelo
        self.model_path = Path(model_path)
        self.drone_label = drone_label

        # Parâmetros de áudio
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_size = int(self.sample_rate * self.block_duration)

        # Parâmetros de decisão
        self.history_size = history_size
        self.avg_threshold = avg_threshold
        self.strong_threshold = strong_threshold

        # Histórico de probabilidades
        self.prob_history = deque(maxlen=self.history_size)

        # Métricas mais recentes
        self.latest_prob: float = 0.0
        self.latest_avg_prob: float = 0.0
        self.latest_status: str = "idle"  # "idle", "running", "error"
        self.last_error_message: str = ""

        # Stream de áudio
        self.stream: sd.InputStream | None = None

        # Carregar modelo treinado
        self._load_model()

    def _load_model(self) -> None:
        """
        Carregar o modelo de classificação salvo em disco.

        Lançar FileNotFoundError se o arquivo não existir.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em: {self.model_path}"
            )
        self.clf = joblib.load(self.model_path)

    def _audio_callback(self, indata, frames, time, status) -> None:
        """
        Callback chamado pelo sounddevice para cada bloco de áudio capturado.

        Parâmetros:
            indata: array numpy [frames, channels] com o bloco de áudio
            frames: quantidade de frames no bloco
            time: informação de tempo (não utilizada aqui)
            status: objeto com flags de underflow/overflow e outros eventos

        Passos:
            1. registrar mensagem de status, se houver
            2. converter áudio para mono
            3. extrair MFCCs do buffer
            4. classificar o vetor de características
            5. atualizar métricas internas (probabilidade e média)
        """
        try:
            if status:
                self.last_error_message = str(status)

            # Converter de [frames, channels] para vetor mono
            audio_mono = indata.mean(axis=1)

            # Extrair MFCC a partir do buffer em memória
            features = extract_mfcc_from_buffer(
                audio_mono,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=1024,
                hop_length=512,
            )

            # Redimensionar para o formato esperado pelo scikit-learn
            X = features.reshape(1, -1)

            # Calcular probabilidade da classe Drone
            if hasattr(self.clf, "predict_proba"):
                proba = self.clf.predict_proba(X)[0]
                classes = self.clf.classes_
                if self.drone_label in classes:
                    idx = list(classes).index(self.drone_label)
                    drone_prob = float(proba[idx])
                else:
                    drone_prob = 0.0
            else:
                pred = self.clf.predict(X)[0]
                drone_prob = 1.0 if pred == self.drone_label else 0.0

            # Atualizar histórico e métricas recentes
            self.prob_history.append(drone_prob)
            avg_prob = sum(self.prob_history) / len(self.prob_history)

            self.latest_prob = drone_prob
            self.latest_avg_prob = avg_prob
            self.latest_status = "running"

        except Exception as exc:
            # Guardar mensagem de erro e marcar estado como erro
            self.last_error_message = str(exc)
            self.latest_status = "error"

    def start(self) -> None:
        """
        Iniciar captura de áudio e processamento em tempo real.

        Criar um InputStream se ainda não existir e iniciar o stream.
        Resetar histórico e métricas ao iniciar.
        """
        if self.stream is not None:
            return

        self.prob_history.clear()
        self.latest_prob = 0.0
        self.latest_avg_prob = 0.0
        self.latest_status = "running"
        self.last_error_message = ""

        # Criar stream de entrada mono com parâmetros configurados
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # Iniciar stream de captura
        self.stream.start()

    def stop(self) -> None:
        """
        Parar captura de áudio e liberar recursos do stream.

        Garantir que o objeto InputStream seja fechado e descartado.
        """
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
                self.latest_status = "idle"

    def is_running(self) -> bool:
        """
        Verificar se o stream de captura está ativo.
        """
        return self.stream is not None

    def get_latest_metrics(self) -> Dict[str, float | str]:
        """
        Retornar dicionário com as métricas mais recentes calculadas.

        Campos retornados:
            prob: probabilidade atual de drone no último bloco
            avg_prob: média das últimas probabilidades
            status: estado atual ("idle", "running", "error")
            error: mensagem de erro mais recente (se houver)
        """
        return {
            "prob": self.latest_prob,
            "avg_prob": self.latest_avg_prob,
            "status": self.latest_status,
            "error": self.last_error_message,
        }
