import numpy as np

class SensorFusion:
    """
    Classe responsável por realizar a fusão de decisão entre os módulos
    de detecção de Radiofrequência (RF) e Acústica.
    Implementa a Fusão de Nível de Decisão Ponderada.
    """
    
    def __init__(self, weight_rf=0.5, weight_acoustic=0.5, decision_threshold=0.6):
        """
        Inicializa o módulo de fusão.
        
        :param weight_rf: Peso dado à probabilidade de detecção por RF.
        :param weight_acoustic: Peso dado à probabilidade de detecção Acústica.
        :param decision_threshold: Limiar para a decisão final (0 a 1).
        """
        if not np.isclose(weight_rf + weight_acoustic, 1.0):
            raise ValueError("A soma dos pesos deve ser igual a 1.0")
            
        self.weight_rf = weight_rf
        self.weight_acoustic = weight_acoustic
        self.decision_threshold = decision_threshold
        
    def fuse_and_decide(self, prob_rf, prob_acoustic):
        """
        Realiza a fusão das probabilidades e toma a decisão final.
        
        :param prob_rf: Probabilidade de detecção de drone pelo módulo RF (0 a 1).
        :param prob_acoustic: Probabilidade de detecção de drone pelo módulo Acústico (0 a 1).
        :return: Uma tupla (final_probability, decision_bool)
        """
        # 1. Fusão Ponderada
        # P_Final = w_RF * P_RF + w_Ac * P_Ac
        final_probability = (self.weight_rf * prob_rf) + (self.weight_acoustic * prob_acoustic)
        
        # 2. Decisão Final
        decision = final_probability >= self.decision_threshold
        
        return final_probability, decision

if __name__ == '__main__':
    # Exemplo de uso:
    
    # Cenário 1: Confirmação Mútua (RF e Acústica concordam)
    fusion_module = SensorFusion(weight_rf=0.5, weight_acoustic=0.5)
    prob_rf_1 = 0.9  # Alta probabilidade por RF
    prob_acoustic_1 = 0.85 # Alta probabilidade por Acústica
    
    prob_final_1, decision_1 = fusion_module.fuse_and_decide(prob_rf_1, prob_acoustic_1)
    
    print("--- Cenário 1: Confirmação Mútua ---")
    print(f"Probabilidade RF: {prob_rf_1:.2f}, Probabilidade Acústica: {prob_acoustic_1:.2f}")
    print(f"Probabilidade Final: {prob_final_1:.2f}, Decisão: {'Drone Detectado' if decision_1 else 'Não Detectado'}")
    
    # Cenário 2: Falso Positivo em uma Modalidade (Acústica detecta, RF não)
    # Exemplo: Ventilador ligado (Alto Acústico) mas sem sinal de controle (Baixo RF)
    prob_rf_2 = 0.1  # Baixa probabilidade por RF
    prob_acoustic_2 = 0.9 # Alta probabilidade por Acústica
    
    prob_final_2, decision_2 = fusion_module.fuse_and_decide(prob_rf_2, prob_acoustic_2)
    
    print("\n--- Cenário 2: Conflito (Falso Positivo Acústico) ---")
    print(f"Probabilidade RF: {prob_rf_2:.2f}, Probabilidade Acústica: {prob_acoustic_2:.2f}")
    print(f"Probabilidade Final: {prob_final_2:.2f}, Decisão: {'Drone Detectado' if decision_2 else 'Não Detectado'}")
    
    # Cenário 3: Ponderação Diferente (RF é mais confiável)
    # Se o ambiente for ruidoso, damos mais peso ao RF
    fusion_module_weighted = SensorFusion(weight_rf=0.8, weight_acoustic=0.2)
    
    prob_final_3, decision_3 = fusion_module_weighted.fuse_and_decide(prob_rf_2, prob_acoustic_2)
    
    print("\n--- Cenário 3: Conflito com Ponderação (RF mais confiável) ---")
    print(f"Pesos: RF=0.8, Acústica=0.2")
    print(f"Probabilidade RF: {prob_rf_2:.2f}, Probabilidade Acústica: {prob_acoustic_2:.2f}")
    print(f"Probabilidade Final: {prob_final_3:.2f}, Decisão: {'Drone Detectado' if decision_3 else 'Não Detectado'}")