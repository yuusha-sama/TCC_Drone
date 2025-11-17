# UI_UX/rf_detector_stub.py
#
# Implementar um stub de detector RF para permitir testar a interface gráfica
# antes da integração com o hardware e com o pipeline RF real.
#
# A ideia é manter a mesma interface pública que o detector acústico:
# - start()
# - stop()
# - is_running()
# - get_latest_metrics()
#
# Quando o módulo RF estiver pronto, substituir a lógica interna desta classe
# por uma implementação real, preservando a assinatura dos métodos.

from typing import Dict


class RFDetectorStub:
    """
    Implementar um detector RF de exemplo, sem acesso real ao hardware.

    Responsabilidades:
    - manter estado ligado/desligado
    - expor métricas em um dicionário compatível com a interface da GUI
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._latest_prob: float = 0.0
        self._latest_status: str = "idle"  # "idle", "running", "error"
        self._last_error_message: str = ""

    def start(self) -> None:
        """
        Iniciar o detector RF (stub).

        Nesta implementação, apenas marcar o estado como "running".
        O detector real deverá iniciar captura e processamento dos dados RF.
        """
        self._running = True
        self._latest_status = "running"
        self._last_error_message = ""

    def stop(self) -> None:
        """
        Parar o detector RF (stub).

        Nesta implementação, apenas marcar o estado como "idle".
        O detector real deverá encerrar captura e liberar recursos.
        """
        self._running = False
        self._latest_status = "idle"

    def is_running(self) -> bool:
        """
        Verificar se o detector RF está ativo.
        """
        return self._running

    def get_latest_metrics(self) -> Dict[str, float | str]:
        """
        Retornar métricas do detector RF.

        Campos retornados:
            prob: probabilidade estimada de presença de drone no canal RF
                  (na versão stub, manter valor fixo, como 0.0)
            status: estado atual ("idle", "running", "error")
            error: mensagem de erro mais recente (se houver)
        """
        return {
            "prob": self._latest_prob,
            "status": self._latest_status,
            "error": self._last_error_message,
        }
