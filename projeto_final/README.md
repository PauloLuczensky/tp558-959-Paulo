# 📘 Projeto TP558 — Federated Learning + XGBoost + Optuna + XAI (Multiclasse)

Este projeto implementa um pipeline completo de **classificação multiclasse** utilizando:

- **Aprendizado Federado (Federated Learning – FL)**
- **XGBoost** como modelo base
- **Optuna** para otimização de hiperparâmetros
- **Técnicas de Explainable AI (XAI)**
- Avaliação, comparação e interpretação dos resultados

O notebook demonstra desde o carregamento do dataset até a explicação dos modelos, sendo ideal para aplicações que exigem privacidade, desempenho otimizado e interpretabilidade.

---

## 📂 Estrutura Geral do Projeto

### 1. 📊 Dataset
Nesta etapa ocorre:
- Carregamento dos dados
- Limpeza e pré-processamento
- Seleção e engenharia de features (quando aplicável)
- Divisão em treino, validação e teste
- Normalização/Padronização dos atributos

---

### 2. 🧠 Treinamento (FL + XGBoost + Optuna)

#### 🔹 Federated Learning
Implementação de aprendizado federado, permitindo treinar modelos em diferentes “clientes” sem compartilhar dados sensíveis.

Fluxo típico:
1. Separação dos dados em múltiplos clientes
2. Treinamento local com XGBoost
3. Agregação dos modelos (ex.: FedAvg)
4. Repetição por várias rodadas federadas

#### 🔹 XGBoost
O modelo de boosting utilizado para classificação multiclasse durante as rodadas locais de aprendizado.

#### 🔹 Optuna — Hyperparameter Tuning
Utilizado para encontrar os melhores hiperparâmetros, como:
- `eta`
- `max_depth`
- `min_child_weight`
- `gamma`
- `subsample`
- `colsample_bytree`
- Entre outros parâmetros do XGBoost

Optuna otimiza automaticamente para maximizar a métrica escolhida (acurácia, F1-score, etc.).

---

### 3. 📈 Resultados
São apresentados:
- Acurácia e métricas por classe
- Matriz de confusão
- Curvas de desempenho (quando aplicável)
- Comparação entre modelo federado e modelo centralizado
- Hiperparâmetros ótimos encontrados pelo Optuna

---

### 4. 🔍 Aplicando XAI (Explainable AI)
Explicações do comportamento do modelo utilizando técnicas como:
- **SHAP values**
- **Feature importance**
- **Summary plots**
- **Decision plots**

Estas explicações permitem interpretar:
- Quais atributos mais influenciam o modelo
- Como as decisões são tomadas para cada classe
- A lógica interna do XGBoost após o treinamento federado

---

## 🚀 Tecnologias Utilizadas

- Python
- XGBoost
- Optuna
- SHAP
- Pandas / NumPy
- Matplotlib / Seaborn
- Framework ou implementação própria de **Federated Learning**

---

## ▶️ Como Executar

1. Instale as dependências:

```bash
pip install xgboost optuna shap pandas numpy matplotlib
```

## 🎯 Objetivo do Projeto

O objetivo é demonstrar como integrar:

- Aprendizado Federado

- Modelagem com XGBoost

- Otimização de hiperparâmetros com Optuna

- Explicabilidade usando XAI em um fluxo robusto de classificação multiclasse capaz de:

    - Preservar privacidade dos dados

    - Maximizar desempenho

    - Aumentar transparência do modelo

## 📝 Autores


Autores :

Alessandra Carolina Domiciano​

Paulo Otavio Luczensky de Souza​
