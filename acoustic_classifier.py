import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class AcousticClassifier:
    """
    Classe responsável por treinar e gerenciar o modelo de classificação
    acústica de drones usando características MFCC e scikit-learn.
    """
    
    def __init__(self, model_path='acoustic_model.joblib'):
        """
        Inicializa o classificador.
        :param model_path: Caminho para salvar/carregar o modelo treinado.
        """
        self.model = None
        self.model_path = model_path
        
    def train_model(self, X: np.ndarray, y: np.ndarray, test_size=0.3, random_state=42):
        """
        Treina um classificador RandomForest com os dados MFCC fornecidos.
        
        :param X: Matriz de características (MFCCs reais).
        :param y: Vetor de rótulos (0 para Não-Drone, 1 para Drone).
        :param test_size: Proporção do dataset a ser usado para teste.
        :param random_state: Semente para reprodutibilidade.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("O número de amostras em X e y deve ser o mesmo.")
            
        print(f"Iniciando treinamento do modelo acústico com {X.shape[0]} amostras.")
        
        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Cria e treina o modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Avalia o modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Treinamento concluído. Acurácia de teste: {accuracy:.4f}")
        
        # Salva o modelo treinado
        self.save_model()
        
    def load_model(self):
        """
        Carrega o modelo treinado do disco.
        """
        try:
            self.model = joblib.load(self.model_path)
            # print(f"Modelo acústico carregado de {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Modelo não encontrado em {self.model_path}. Necessário treinar.")
            return False

    def save_model(self):
        """
        Salva o modelo treinado no disco.
        """
        if self.model:
            joblib.dump(self.model, self.model_path)
            # print(f"Modelo acústico salvo em {self.model_path}")
        
    def predict(self, mfcc_features: np.ndarray) -> float:
        """
        Realiza a previsão de detecção de drone.
        
        :param mfcc_features: Um array 2D de MFCCs (1, N_MFCC) de um segmento de áudio.
        :return: Probabilidade de ser um drone (float entre 0 e 1).
        """
        if self.model is None:
            if not self.load_model():
                # Se o modelo não carregar, retorna 0.5 (indecisão) para evitar quebrar o sistema
                return 0.5 
        
        # O predict_proba retorna as probabilidades para cada classe [prob_non_drone, prob_drone]
        probabilities = self.model.predict_proba(mfcc_features.reshape(1, -1))
        
        # Retorna a probabilidade de ser a classe 1 (Drone)
        prob_drone = probabilities[0][1]
        
        return prob_drone

if __name__ == '__main__':
    # Exemplo de uso com dados simulados para teste (Apenas para garantir que o código funcione)
    N_MFCC = 20
    N_SAMPLES = 200
    
    # Simulação de dados reais de treinamento
    mfcc_drone = np.random.normal(loc=5, scale=2, size=(N_SAMPLES // 2, N_MFCC))
    labels_drone = np.ones(N_SAMPLES // 2)
    mfcc_non_drone = np.random.normal(loc=0, scale=3, size=(N_SAMPLES // 2, N_MFCC))
    labels_non_drone = np.zeros(N_SAMPLES // 2)
    
    X_train_data = np.vstack((mfcc_drone, mfcc_non_drone))
    y_train_labels = np.hstack((labels_drone, labels_non_drone))
    
    classifier = AcousticClassifier()
    classifier.train_model(X_train_data, y_train_labels)
    
    # Simulação de previsão
    new_mfcc_drone = np.random.normal(loc=5, scale=2, size=(N_MFCC,))
    prob_drone = classifier.predict(new_mfcc_drone)
    print(f"Previsão (Drone): Probabilidade = {prob_drone:.4f}")