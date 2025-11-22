# sensor_fusion_core.py
#
# Implementar o motor de fusão de sensores em nível de decisão.
# A ideia é combinar as probabilidades vindas dos detectores acústico e RF
# em uma probabilidade única de "presença de drone" e decidir se há drone.

from typing import Dict, Optional


class SensorFusionEngine:
    def __init__(
        self,
        weight_acoustic: float = 0.5,
        weight_rf: float = 0.5,
        fusion_threshold: float = 0.5,
    ) -> None:
        # Pesos relativos de cada canal na fusão
        self.weight_acoustic = weight_acoustic
        self.weight_rf = weight_rf

        # Limiar de decisão sobre a probabilidade fundida
        self.fusion_threshold = fusion_threshold

        # Métricas mais recentes
        self.latest_fused_prob: float = 0.0
        self.latest_status: str = "idle"  # "idle", "running", "error"
        self.last_error_message: str = ""
        self.last_source: str = "none"  # "none", "acoustic", "rf", "fusion"

    def update(
        self,
        acoustic_prob: Optional[float],
        rf_prob: Optional[float],
        acoustic_active: bool,
        rf_active: bool,
    ) -> None:
        
        try:
            if not acoustic_active and not rf_active:
                self.latest_fused_prob = 0.0
                self.latest_status = "idle"
                self.last_source = "none"
                return

            if acoustic_active and not rf_active and acoustic_prob is not None:
                self.latest_fused_prob = float(acoustic_prob)
                self.latest_status = "running"
                self.last_source = "acoustic"
                return

            if rf_active and not acoustic_active and rf_prob is not None:
                self.latest_fused_prob = float(rf_prob)
                self.latest_status = "running"
                self.last_source = "rf"
                return

            # Ambos os canais ativos
            if acoustic_prob is None:
                acoustic_prob = 0.0
            if rf_prob is None:
                rf_prob = 0.0

            total_weight = self.weight_acoustic + self.weight_rf
            if total_weight <= 0.0:
                # Evitar divisão por zero; cair em média simples
                self.latest_fused_prob = (
                    float(acoustic_prob) + float(rf_prob)
                ) / 2.0
            else:
                # Média ponderada das probabilidades
                self.latest_fused_prob = (
                    self.weight_acoustic * float(acoustic_prob)
                    + self.weight_rf * float(rf_prob)
                ) / total_weight

            self.latest_status = "running"
            self.last_source = "fusion"

        except Exception as exc:
            self.last_error_message = str(exc)
            self.latest_status = "error"

    def get_latest_metrics(self) -> Dict[str, float | str]:
        return {
            "fused_prob": self.latest_fused_prob,
            "status": self.latest_status,
            "source": self.last_source,
            "error": self.last_error_message,
        }
