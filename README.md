# ❤️ Heart Attack Prediction App

Uma aplicação web interativa para predição de risco de ataque cardíaco utilizando Machine Learning, construída com Streamlit e Scikit-learn.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Sobre o Projeto

Este projeto implementa um modelo de machine learning para prever o risco de ataque cardíaco com base em características médicas do paciente. A aplicação oferece uma interface amigável onde usuários podem inserir dados e receber previsões em tempo real.

### 🎯 Funcionalidades

- ✅ **Predição em tempo real** do risco de ataque cardíaco
- ✅ **Interface intuitiva** com formulários interativos
- ✅ **Análise exploratória** dos dados com visualizações
- ✅ **Múltiplos modelos** de machine learning
- ✅ **Explicabilidade** das previsões com SHAP
- ✅ **Dashboard** com métricas de performance

## 🛠️ Tecnologias Utilizadas

- **Python 3.7+**
- **Streamlit** - Framework para aplicações web
- **Scikit-learn** - Machine Learning
- **Pandas & NumPy** - Manipulação de dados
- **Matplotlib & Seaborn** - Visualizações
- **SHAP** - Explicabilidade do modelo
- **Joblib** - Serialização do modelo

## 🚀 Como Executar

### Pré-requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes do Python)

### 📥 Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/jonatan777/heart_prediction_app.git
cd heart_prediction_app

### 2. Crie um ambiente virtual (recomendado):

bash

# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/Mac
python3 -m venv venv
source venv/bin/activate

3. Instale as dependências:

bash

pip install -r requirements.txt

Caso o arquivo requirements.txt não exista, instale as dependências manualmente:
bash

pip install streamlit scikit-learn pandas numpy matplotlib seaborn shap joblib

4. 🎮 Executando a Aplicação:

    Navegue até o diretório do projeto:

bash

cd heart_prediction_app

    Execute o comando do Streamlit:

bash

streamlit run heart_attack_improved.py

6. Acesse a aplicação:

text

http://localhost:8501

A aplicação abrirá automaticamente no seu navegador padrão.
📁 Estrutura do Projeto
text

heart_prediction_app/
│
├── data
│   └── heart.csv                # Dataset utilizado (se disponível)
├── heart_attack_improved.py     # Código principal da aplicação                 
├── requirements.txt             # Dependências do projeto
└── README.md


🎯 Como Usar
1. Página Inicial

    Visão geral do projeto

    Estatísticas do dataset

    Informações sobre a importância do projeto

2. Predição

    Preencha o formulário com dados do paciente:

        Idade, sexo, tipo de dor no peito

        Pressão arterial, colesterol, açúcar no sangue

        Eletrocardiograma, frequência cardíaca máxima

        Angina induzida por exercício, etc.

3. Resultados

    Probabilidade de risco (0-100%)

    Classificação (Baixo/Alto Risco)

    Explicação das features mais importantes

    Recomendações baseadas no resultado

4. Análise dos Dados

    Visualizações interativas

    Correlação entre features

    Distribuição dos dados

    Performance do modelo

🏗️ Arquitetura do Modelo
🔍 Pré-processamento

    Tratamento de valores missing

    Normalização de features

    Encoding de variáveis categóricas

🤖 Modelos Implementados

    Random Forest Classifier

    Gradient Boosting

    Support Vector Machine

    Logistic Regression

    K-Nearest Neighbors

📊 Métricas de Avaliação

    Acurácia

    Precisão

    Recall

    F1-Score

    Matriz de Confusão

    Curva ROC

🎛️ Parâmetros do Modelo
Random Forest (Modelo Principal)
python

RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

📈 Dataset

O projeto utiliza o dataset Heart Attack Analysis & Prediction contendo:

    303 instâncias

    14 features por instância

    Variável alvo: Risco de ataque cardíaco (0 = Baixo, 1 = Alto)

Principais Features:

    age: Idade do paciente

    sex: Sexo (1 = masculino, 0 = feminino)

    cp: Tipo de dor no peito

    trestbps: Pressão arterial em repouso

    chol: Colesterol sérico

    fbs: Açúcar no sangue em jejum

    thalach: Frequência cardíaca máxima

    exang: Angina induzida por exercício

🔧 Desenvolvimento
Adicionando Novos Modelos
python

# No código, adicione novos modelos no dicionário:
novos_modelos = {
    'Novo Modelo': SeuModeloClass()
}

Customizando a Interface

Edite as seções do Streamlit no arquivo heart_attack_improved.py:
python

st.title("Seu Novo Título")
st.sidebar.selectbox("Nova Opção", opcoes)

🐛 Solução de Problemas
Erro: "ModuleNotFoundError"
bash

# Certifique-se que todas as dependências estão instaladas
pip install --upgrade -r requirements.txt

Erro: "Address already in use"
bash

# Use uma porta diferente
streamlit run heart_attack_improved.py --server.port 8502

Aplicação não carrega

    Verifique se o arquivo heart.csv está no diretório correto

    Confirme que o Python 3.7+ está instalado

🤝 Contribuindo

Contribuições são bem-vindas! Siga estos passos:

    Fork o projeto

    Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)

    Commit suas mudanças (git commit -m 'Add some AmazingFeature')

    Push para a branch (git push origin feature/AmazingFeature)

    Abra um Pull Request

📄 Licença

Distribuído sob licença MIT. Veja LICENSE para mais informações.
👨‍💻 Autor

Jonatan

    GitHub: @jonatan777

    Projeto: Heart Prediction App

🙏 Agradecimentos

    Dataset: Heart Attack Analysis & Prediction Dataset

    Streamlit team pela excelente framework

    Comunidade Scikit-learn

⚠️ Aviso Legal: Esta aplicação é para fins educacionais e de demonstração. Não substitui aconselhamento médico profissional. Sempre consulte um médico para questões de saúde.**
text
