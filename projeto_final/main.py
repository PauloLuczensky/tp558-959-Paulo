# ==============================================================================
# IMPORTS
# ==============================================================================
import numpy as np  # Operações numéricas e arrays
import xgboost as xgb  # Biblioteca XGBoost para gradient boosting
from sklearn.datasets import make_classification  # Geração de dados sintéticos
from sklearn.model_selection import train_test_split  # Split de dados
from sklearn.metrics import accuracy_score, classification_report  # Métricas
import pickle  # Serialização de objetos Python
import copy  # Cópia profunda de objetos
from typing import List, Dict, Tuple  # Type hints para melhor documentação
import json  # Manipulação de JSON

# ==============================================================================
# CLASSE: FEDERATED CLIENT
# ==============================================================================
class FederatedClient:
    """
    Cliente para Federated Learning com XGBoost
    
    Cada cliente representa um nó independente que:
    - Possui seus próprios dados locais (privados)
    - Treina um modelo localmente sem compartilhar dados brutos
    - Envia apenas o modelo treinado (parâmetros) para o servidor
    """
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray):
        """
        Inicializa um cliente federado
        
        Args:
            client_id: Identificador único do cliente
            X_train: Features de treinamento (dados locais do cliente)
            y_train: Labels de treinamento (dados locais do cliente)
            X_test: Features de teste (dados locais do cliente)
            y_test: Labels de teste (dados locais do cliente)
        """
        self.client_id = client_id  # ID único para identificar o cliente
        self.X_train = X_train  # Dados de treino privados do cliente
        self.y_train = y_train  # Labels de treino privados
        self.X_test = X_test  # Dados de teste privados
        self.y_test = y_test  # Labels de teste privados
        self.model = None  # Modelo XGBoost local (inicialmente vazio)
        
    def train_local_model(self, global_model_params: Dict = None, 
                         xgb_params: Dict = None, num_rounds: int = 10) -> Dict:
        """
        Treina modelo local com dados do cliente
        
        Este é o coração do treinamento federado. Cada cliente:
        1. Recebe o modelo global atual (se existir)
        2. Treina localmente com seus próprios dados
        3. Retorna o modelo atualizado (sem compartilhar dados)
        
        Args:
            global_model_params: Parâmetros do modelo global serializado (bytes)
                                Se None, treina do zero
            xgb_params: Hiperparâmetros do XGBoost (dict)
                       Se None, usa configuração padrão
            num_rounds: Número de rounds de boosting (árvores a adicionar)
            
        Returns:
            Dicionário contendo:
                - client_id: ID do cliente
                - model: Modelo treinado serializado (bytes)
                - accuracy: Acurácia no conjunto de teste local
                - num_samples: Número de amostras de treino (para ponderação)
        """
        # Define hiperparâmetros padrão se não fornecidos
        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',  # Classificação binária
                'max_depth': 5,  # Profundidade máxima das árvores
                'eta': 0.1,  # Taxa de aprendizado (learning rate)
                'subsample': 0.8,  # Proporção de amostras usadas por árvore
                'colsample_bytree': 0.8,  # Proporção de features por árvore
                'eval_metric': 'logloss'  # Métrica de avaliação
            }
        
        # Converte dados para formato DMatrix (otimizado para XGBoost)
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        # Verifica se existe modelo global para continuar treinamento
        if global_model_params:
            # APRENDIZADO FEDERADO: Carrega modelo global como ponto de partida
            self.model = pickle.loads(global_model_params)
            # Continua treinamento a partir do modelo global
            # Isso adiciona novas árvores ao modelo existente
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_rounds,  # Adiciona num_rounds novas árvores
                xgb_model=self.model,  # Usa modelo global como base
                evals=[(dtest, 'test')],  # Monitora desempenho no teste
                verbose_eval=False  # Não imprime logs detalhados
            )
        else:
            # PRIMEIRA RODADA: Treina modelo do zero
            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=num_rounds,  # Cria num_rounds árvores
                evals=[(dtest, 'test')],
                verbose_eval=False
            )
        
        # Avalia modelo local no conjunto de teste
        y_pred = self.model.predict(dtest)  # Predições (probabilidades)
        y_pred_binary = (y_pred > 0.5).astype(int)  # Converte para classes binárias
        accuracy = accuracy_score(self.y_test, y_pred_binary)  # Calcula acurácia
        
        # Serializa modelo para envio ao servidor
        # IMPORTANTE: Apenas o modelo é enviado, NUNCA os dados brutos
        model_bytes = pickle.dumps(self.model)
        
        # Retorna update para o servidor
        return {
            'client_id': self.client_id,
            'model': model_bytes,  # Modelo serializado
            'accuracy': accuracy,  # Métrica de desempenho
            'num_samples': len(self.X_train)  # Para agregação ponderada
        }
    
    def evaluate(self) -> Dict:
        """
        Avalia modelo local no conjunto de teste
        
        Útil para verificar o desempenho do modelo depois do treinamento
        
        Returns:
            Dicionário com métricas de avaliação
        """
        # Verifica se o modelo foi treinado
        if self.model is None:
            return {'error': 'Modelo não treinado'}
        
        # Prepara dados de teste
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        
        # Faz predições
        y_pred = self.model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Retorna métricas
        return {
            'client_id': self.client_id,
            'accuracy': accuracy_score(self.y_test, y_pred_binary),
            'num_samples': len(self.X_test)
        }


# ==============================================================================
# CLASSE: FEDERATED SERVER
# ==============================================================================
class FederatedServer:
    """
    Servidor central para coordenar Federated Learning
    
    O servidor é responsável por:
    - Receber modelos treinados dos clientes
    - Agregar os modelos em um modelo global
    - Distribuir o modelo global atualizado de volta aos clientes
    - Coordenar rounds de treinamento
    - Avaliar desempenho global
    
    IMPORTANTE: O servidor NUNCA tem acesso aos dados brutos dos clientes!
    """
    
    def __init__(self, aggregation_strategy: str = 'weighted_average'):
        """
        Inicializa o servidor federado
        
        Args:
            aggregation_strategy: Estratégia para combinar modelos dos clientes
                - 'weighted_average': Pondera por número de amostras (recomendado)
                - 'simple_average': Média simples sem ponderação
        """
        self.global_model = None  # Modelo global (agregado de todos os clientes)
        self.aggregation_strategy = aggregation_strategy  # Estratégia de agregação
        
        # Histórico de treinamento para análise posterior
        self.history = {
            'rounds': [],  # Número do round
            'global_accuracy': [],  # Acurácia do modelo global
            'client_accuracies': []  # Acurácias individuais dos clientes
        }
        
    def aggregate_models(self, client_updates: List[Dict]) -> bytes:
        """
        Agrega modelos dos clientes em um modelo global
        
        Esta é a etapa central do Federated Learning onde os modelos
        individuais são combinados em um único modelo global.
        
        Args:
            client_updates: Lista de dicionários, cada um contendo:
                - 'model': Modelo do cliente serializado
                - 'num_samples': Número de amostras usadas no treino
                - 'accuracy': Acurácia local
                
        Returns:
            Modelo global serializado (bytes) para distribuir aos clientes
        """
        # Seleciona estratégia de agregação
        if self.aggregation_strategy == 'weighted_average':
            return self._weighted_average_aggregation(client_updates)
        elif self.aggregation_strategy == 'simple_average':
            return self._simple_average_aggregation(client_updates)
        else:
            raise ValueError(f"Estratégia desconhecida: {self.aggregation_strategy}")
    
    def _weighted_average_aggregation(self, client_updates: List[Dict]) -> bytes:
        """
        Agrega modelos usando média ponderada pelo número de amostras
        
        Clientes com mais dados têm maior influência no modelo global.
        Esta é a estratégia recomendada para Federated Learning pois:
        - Reflete melhor a distribuição real dos dados
        - Clientes com mais dados contribuem mais
        - Funciona bem com dados heterogêneos (Non-IID)
        
        NOTA: Esta é uma implementação simplificada. Em produção, seria
        necessário agregar os pesos individuais das árvores do XGBoost.
        Aqui, usamos o modelo do cliente com mais dados como base.
        
        Args:
            client_updates: Lista com updates dos clientes
            
        Returns:
            Modelo global serializado
        """
        # Calcula pesos baseados no número de amostras de cada cliente
        total_samples = sum(update['num_samples'] for update in client_updates)
        weights = [update['num_samples'] / total_samples for update in client_updates]
        
        # Desserializa todos os modelos recebidos
        models = [pickle.loads(update['model']) for update in client_updates]
        
        # Estratégia simplificada: usa o modelo do cliente com mais dados
        # Em uma implementação completa, seria necessário:
        # 1. Extrair pesos de cada árvore de cada modelo
        # 2. Fazer média ponderada dos pesos
        # 3. Reconstruir modelo com pesos agregados
        best_client_idx = np.argmax([update['num_samples'] for update in client_updates])
        self.global_model = models[best_client_idx]
        
        # Serializa e retorna modelo global
        return pickle.dumps(self.global_model)
    
    def _simple_average_aggregation(self, client_updates: List[Dict]) -> bytes:
        """
        Agrega modelos usando média simples (sem ponderação)
        
        Todos os clientes têm igual influência, independente do número de amostras.
        Útil quando:
        - Todos os clientes têm aproximadamente a mesma quantidade de dados
        - Queremos evitar viés de clientes com muito mais dados
        
        Args:
            client_updates: Lista com updates dos clientes
            
        Returns:
            Modelo global serializado
        """
        # Desserializa modelos
        models = [pickle.loads(update['model']) for update in client_updates]
        
        # Estratégia simplificada: usa o primeiro modelo como referência
        # Em produção, faria média real dos pesos de todas as árvores
        self.global_model = models[0]
        
        return pickle.dumps(self.global_model)
    
    def federated_round(self, clients: List[FederatedClient], 
                       xgb_params: Dict = None, 
                       num_local_rounds: int = 10) -> Dict:
        """
        Executa uma rodada completa de Federated Learning
        
        Um round completo consiste em:
        1. Servidor envia modelo global atual para todos os clientes
        2. Cada cliente treina localmente com seus dados
        3. Clientes enviam modelos atualizados de volta ao servidor
        4. Servidor agrega os modelos em novo modelo global
        
        Este é o ciclo fundamental do Federated Learning!
        
        Args:
            clients: Lista de clientes participantes
            xgb_params: Hiperparâmetros do XGBoost
            num_local_rounds: Número de rounds de treinamento em cada cliente
            
        Returns:
            Dicionário com métricas da rodada:
                - client_accuracies: Lista de acurácias de cada cliente
                - avg_accuracy: Acurácia média
                - num_clients: Número de clientes participantes
        """
        # PASSO 1: Prepara modelo global para distribuição
        # Serializa modelo global se existir (para enviar aos clientes)
        global_model_params = None
        if self.global_model is not None:
            global_model_params = pickle.dumps(self.global_model)
        
        # PASSO 2: Treinamento local em cada cliente
        client_updates = []  # Armazenará updates de todos os clientes
        client_accuracies = []  # Armazenará acurácias locais
        
        print(f"\n{'='*60}")
        print("TREINAMENTO LOCAL DOS CLIENTES")
        print(f"{'='*60}")
        
        # Cada cliente treina independentemente
        for client in clients:
            print(f"\nCliente {client.client_id} treinando...")
            
            # Cliente treina com seus dados locais
            # IMPORTANTE: Dados NUNCA saem do cliente!
            update = client.train_local_model(
                global_model_params=global_model_params,  # Modelo global como base
                xgb_params=xgb_params,
                num_rounds=num_local_rounds
            )
            
            # Coleta updates (apenas modelos, não dados!)
            client_updates.append(update)
            client_accuracies.append(update['accuracy'])
            
            # Mostra progresso
            print(f"  Acurácia local: {update['accuracy']:.4f}")
            print(f"  Amostras de treino: {update['num_samples']}")
        
        # PASSO 3: Agregação no servidor
        print(f"\n{'='*60}")
        print("AGREGAÇÃO NO SERVIDOR")
        print(f"{'='*60}")
        print(f"Estratégia: {self.aggregation_strategy}")
        
        # Combina modelos dos clientes em modelo global
        self.global_model = pickle.loads(self.aggregate_models(client_updates))
        
        # PASSO 4: Calcula métricas da rodada
        avg_accuracy = np.mean(client_accuracies)
        
        print(f"Acurácia média dos clientes: {avg_accuracy:.4f}")
        
        return {
            'client_accuracies': client_accuracies,
            'avg_accuracy': avg_accuracy,
            'num_clients': len(clients)
        }
    
    def evaluate_global_model(self, clients: List[FederatedClient]) -> Dict:
        """
        Avalia modelo global em todos os clientes
        
        Testa o modelo global nos dados de teste de cada cliente.
        Isso simula o desempenho em dados distribuídos que o modelo
        nunca viu durante o treinamento.
        
        Args:
            clients: Lista de clientes para avaliação
            
        Returns:
            Dicionário com métricas globais:
                - global_accuracy: Acurácia média em todos os clientes
                - client_accuracies: Lista de acurácias individuais
                - total_test_samples: Total de amostras testadas
        """
        # Verifica se existe modelo global
        if self.global_model is None:
            return {'error': 'Modelo global não existe'}
        
        accuracies = []  # Acurácias em cada cliente
        total_samples = 0  # Contador total de amostras testadas
        
        print(f"\n{'='*60}")
        print("AVALIAÇÃO DO MODELO GLOBAL")
        print(f"{'='*60}")
        
        # Testa modelo global em cada cliente
        for client in clients:
            # Prepara dados de teste do cliente
            dtest = xgb.DMatrix(client.X_test, label=client.y_test)
            
            # Faz predições com modelo global
            y_pred = self.global_model.predict(dtest)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calcula acurácia
            acc = accuracy_score(client.y_test, y_pred_binary)
            accuracies.append(acc)
            total_samples += len(client.X_test)
            
            # Mostra resultado para este cliente
            print(f"\nCliente {client.client_id}:")
            print(f"  Acurácia: {acc:.4f}")
            print(f"  Amostras de teste: {len(client.X_test)}")
        
        # Calcula acurácia global (média)
        global_accuracy = np.mean(accuracies)
        
        print(f"\n{'='*60}")
        print(f"ACURÁCIA GLOBAL MÉDIA: {global_accuracy:.4f}")
        print(f"{'='*60}")
        
        return {
            'global_accuracy': global_accuracy,
            'client_accuracies': accuracies,
            'total_test_samples': total_samples
        }
    
def create_federated_data(n_clients: int = 5, n_samples: int = 10000, 
                         n_features: int = 20, test_size: float = 0.2,
                         non_iid: bool = False) -> Tuple[List, List]:
    """
    Cria dados simulados para Federated Learning
    
    Args:
        n_clients: Número de clientes
        n_samples: Número total de amostras
        n_features: Número de features
        test_size: Proporção de dados para teste
        non_iid: Se True, distribui dados de forma não-IID entre clientes
        
    Returns:
        Tupla com listas de dados de treino e teste por cliente
    """
    print(f"\n{'='*60}")
    print("CRIAÇÃO DOS DADOS FEDERADOS")
    print(f"{'='*60}")
    print(f"Número de clientes: {n_clients}")
    print(f"Total de amostras: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Distribuição: {'Non-IID' if non_iid else 'IID'}")
    
    # Gera dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split treino/teste global
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if non_iid:
        # Distribui dados de forma não-IID (cada cliente tem mais exemplos de uma classe)
        indices_class_0 = np.where(y_train == 0)[0]
        indices_class_1 = np.where(y_train == 1)[0]
        
        # Embaralha
        np.random.shuffle(indices_class_0)
        np.random.shuffle(indices_class_1)
        
        # Divide com desbalanceamento
        client_train_data = []
        samples_per_client = len(y_train) // n_clients
        
        for i in range(n_clients):
            # Proporção variável entre classes
            ratio = 0.7 if i % 2 == 0 else 0.3
            n_class_0 = int(samples_per_client * ratio)
            n_class_1 = samples_per_client - n_class_0
            
            start_0 = i * n_class_0
            end_0 = start_0 + n_class_0
            start_1 = i * n_class_1
            end_1 = start_1 + n_class_1
            
            idx_0 = indices_class_0[start_0:end_0] if end_0 <= len(indices_class_0) else indices_class_0[start_0:]
            idx_1 = indices_class_1[start_1:end_1] if end_1 <= len(indices_class_1) else indices_class_1[start_1:]
            
            client_indices = np.concatenate([idx_0, idx_1])
            client_train_data.append(client_indices)
    else:
        # Distribui dados de forma IID
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        client_train_data = np.array_split(indices, n_clients)
    
    # Distribui dados de teste igualmente
    test_indices = np.arange(len(y_test))
    np.random.shuffle(test_indices)
    client_test_data = np.array_split(test_indices, n_clients)
    
    # Mostra distribuição
    print(f"\nDistribuição dos dados:")
    for i in range(n_clients):
        train_idx = client_train_data[i]
        test_idx = client_test_data[i]
        print(f"  Cliente {i}: {len(train_idx)} treino, {len(test_idx)} teste")
    
    return client_train_data, client_test_data, X_train, y_train, X_test, y_test


def main():
    """Função principal para demonstrar o sistema"""
    
    print("\n" + "="*60)
    print("SISTEMA DE FEDERATED LEARNING COM XGBOOST")
    print("="*60)
    
    # Configurações
    N_CLIENTS = 5
    N_SAMPLES = 10000
    N_FEATURES = 20
    N_ROUNDS = 5
    LOCAL_ROUNDS = 10
    NON_IID = True  # Mude para False para distribuição IID
    
    # Cria dados federados
    client_train_indices, client_test_indices, X_train, y_train, X_test, y_test = create_federated_data(
        n_clients=N_CLIENTS,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        non_iid=NON_IID
    )
    
    # Cria clientes
    clients = []
    for i in range(N_CLIENTS):
        train_idx = client_train_indices[i]
        test_idx = client_test_indices[i]
        
        client = FederatedClient(
            client_id=i,
            X_train=X_train[train_idx],
            y_train=y_train[train_idx],
            X_test=X_test[test_idx],
            y_test=y_test[test_idx]
        )
        clients.append(client)
    
    # Cria servidor
    server = FederatedServer(aggregation_strategy='weighted_average')
    
    # Parâmetros XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'seed': 42
    }
    
    # Executa rounds de Federated Learning
    print(f"\n{'='*60}")
    print(f"INICIANDO {N_ROUNDS} ROUNDS DE FEDERATED LEARNING")
    print(f"{'='*60}")
    
    for round_num in range(N_ROUNDS):
        print(f"\n{'#'*60}")
        print(f"ROUND {round_num + 1}/{N_ROUNDS}")
        print(f"{'#'*60}")
        
        # Executa round
        round_metrics = server.federated_round(
            clients=clients,
            xgb_params=xgb_params,
            num_local_rounds=LOCAL_ROUNDS
        )
        
        # Avalia modelo global
        global_metrics = server.evaluate_global_model(clients)
        
        # Armazena histórico
        server.history['rounds'].append(round_num + 1)
        server.history['global_accuracy'].append(global_metrics['global_accuracy'])
        server.history['client_accuracies'].append(round_metrics['client_accuracies'])
    
    # Resultados finais
    print(f"\n{'='*60}")
    print("RESULTADOS FINAIS")
    print(f"{'='*60}")
    print(f"\nHistórico de acurácia global:")
    for i, acc in enumerate(server.history['global_accuracy']):
        print(f"  Round {i+1}: {acc:.4f}")
    
    print(f"\nMelhoria total: {server.history['global_accuracy'][-1] - server.history['global_accuracy'][0]:.4f}")
    print(f"Acurácia final: {server.history['global_accuracy'][-1]:.4f}")
    
    return server, clients


if __name__ == "__main__":
    server, clients = main()