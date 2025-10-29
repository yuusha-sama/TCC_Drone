import numpy as np
import time
import os
import joblib

# Importando as classes integradas
from acoustic_classifier import AcousticClassifier
from sensor_fusion import SensorFusion
from rf_locator_classes import RTL_SDR_Acquisition, RF_Locator
from acoustic_mfcc_classes import Audio_Acquisition, MFCC_Extractor

# --- Função Principal ---
def run_lpccd_system(rf_frequency=433e6):
    """
    Orquestra o sistema LPCDD, executando a detecção por RF e Acústica,
    e realizando a fusão de sensores.
    """
    
    # 1. Inicialização dos Módulos
    print("--- Inicializando LPCDD ---")
    
    # Módulos de RF
    rtl_sdr_acq = RTL_SDR_Acquisition()
    rf_locator = RF_Locator(rtl_sdr_acq)
    
    # Módulos Acústicos
    audio_acq = Audio_Acquisition()
    mfcc_extractor = MFCC_Extractor()
    acoustic_classifier = AcousticClassifier()
    
    # Módulo de Fusão (Pesos podem ser ajustados com base em calibração real)
    fusion_module = SensorFusion(weight_rf=0.6, weight_acoustic=0.4) # Exemplo: dando mais peso ao RF
    
    # 2. Treinamento (Verificação)
    # O modelo acústico deve ser treinado antes de rodar o sistema.
    # Esta seção é apenas para demonstração:
    if not acoustic_classifier.load_model():
        print("\n[AVISO]: Modelo Acústico não encontrado. Treinamento simulado necessário.")
        # Simulação de dados para TREINAMENTO, pois o usuário não forneceu os dados reais
        N_MFCC = 20
        N_SAMPLES = 200
        mfcc_drone = np.random.normal(loc=5, scale=2, size=(N_SAMPLES // 2, N_MFCC))
        labels_drone = np.ones(N_SAMPLES // 2)
        mfcc_non_drone = np.random.normal(loc=0, scale=3, size=(N_SAMPLES // 2, N_MFCC))
        labels_non_drone = np.zeros(N_SAMPLES // 2)
        X_train_data = np.vstack((mfcc_drone, mfcc_non_drone))
        y_train_labels = np.hstack((labels_drone, labels_non_drone))
        acoustic_classifier.train_model(X_train_data, y_train_labels)
        print("[AVISO]: Treinamento simulado concluído. Substitua por dados reais.")
        
    print("\n--- Iniciando Loop de Detecção em Tempo Real ---")
    
    # Simulação de um loop de detecção contínua
    for i in range(5):
        print(f"\n--- Iteração {i+1} ---")
        
        # --- A. Módulo Acústico (Extração e Previsão) ---
        # No loop real, você chamaria audio_acq.record_audio()
        # Para evitar travar o sistema, vamos simular a extração de MFCC de um segmento
        # Assumindo que o MFCC_Extractor retorna um array 1D de 20 features
        
        # Simulação de um cenário misto (para demonstrar a fusão)
        if i % 2 == 0:
             # Sinal forte (Simula drone presente)
            sim_mfcc = np.random.normal(loc=5, scale=1, size=(mfcc_extractor.n_mfcc,))
        else:
            # Sinal fraco (Simula ruído)
            sim_mfcc = np.random.normal(loc=0, scale=1, size=(mfcc_extractor.n_mfcc,))
            
        mfcc_features = sim_mfcc
        prob_acoustic = acoustic_classifier.predict(mfcc_features)
        
        # --- B. Módulo RF (Aquisição e Previsão) ---
        prob_rf = rf_locator.get_detection_probability(center_freq=rf_frequency)
        
        print(f"Probabilidade Acústica (IA): {prob_acoustic:.4f}")
        print(f"Probabilidade RF (RSSI Regressão): {prob_rf:.4f}")
        
        # --- C. Módulo de Fusão ---
        prob_final, decision = fusion_module.fuse_and_decide(prob_rf, prob_acoustic)
        
        print(f"Probabilidade Final (Fusão): {prob_final:.4f}")
        
        if decision:
            print(">>> DRONE DETECTADO E CONFIRMADO PELA FUSÃO <<<")
            
            # --- D. Localização do Piloto (Triangulação) ---
            # Para a triangulação, você precisaria de múltiplas medições de RSSI e azimute.
            # Aqui, simulamos o status de localização baseado na confirmação.
            print(rf_locator.locate_pilot_triangulation([(0, -50)])) # Simula uma medição para DoA
        else:
            print("--- Nenhuma detecção confirmada ---")
            
        time.sleep(0.5) # Pequeno delay

if __name__ == '__main__':
    # É necessário instalar as dependências: rtlsdr, librosa, sounddevice, scikit-learn, joblib
    # O RTL-SDR real precisa de drivers instalados no seu sistema.
    # O sounddevice pode precisar de configuração de microfone.
    
    try:
        run_lpccd_system(rf_frequency=433e6)
    except Exception as e:
        print(f"\n[ERRO FATAL]: Ocorreu um erro durante a execução do sistema. Verifique as dependências de hardware (RTL-SDR, Microfone) e software (librosa, rtlsdr, sounddevice). Erro: {e}")
        # Se o erro for devido ao RTL-SDR não estar conectado, ele deve cair no modo de simulação interna.
        # Se o erro for de dependência, o usuário deve instalar as bibliotecas.