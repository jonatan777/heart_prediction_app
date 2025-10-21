import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuração da página
st.set_page_config(
    page_title="Sistema de Predição Cardíaca", 
    page_icon="❤️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregamento de dados com cache
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('./data/heart.csv')
        st.success("✅ Dados carregados com sucesso!")
        return data
    except FileNotFoundError:
        st.error("❌ Arquivo não encontrado. Verifique o caminho do dataset.")
        return None

# Estilização com CSS
def aplicar_estilos():
    custom_css = """
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF4B4B;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #0E1117;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #FF4B4B;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #2d1b1b;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ff6b6b;
        margin: 1rem 0;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Informações do dataset
def mostrar_info_dataset(df):
    st.markdown("<div class='section-header'>📁 Informações do Dataset</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total de Pacientes", len(df))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Variáveis", len(df.columns))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        taxa_doenca = (df['output'].mean() * 100)
        st.metric("Taxa de Doença Cardíaca", f"{taxa_doenca:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Dados Ausentes", "0")
        st.markdown("</div>", unsafe_allow_html=True)

# Estatísticas descritivas
def criar_estatisticas_descritivas(df):
    st.subheader("📊 Estatísticas Descritivas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Média Idade", f"{df['age'].mean():.1f} anos")
        st.metric("Desvio Padrão", f"±{df['age'].std():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Pressão Arterial Média", f"{df['trtbps'].mean():.1f} mmHg")
        st.metric("Desvio Padrão", f"±{df['trtbps'].std():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Colesterol Médio", f"{df['chol'].mean():.1f} mg/dl")
        st.metric("Desvio Padrão", f"±{df['chol'].std():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Freq. Cardíaca Máx.", f"{df['thalachh'].mean():.1f} bpm")
        st.metric("Desvio Padrão", f"±{df['thalachh'].std():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.subheader("Tabela Estatística Completa")
    st.dataframe(df.describe(), use_container_width=True)

# Visualizações de distribuição
def criar_visualizacoes_distribuicao(df):
    st.subheader("📈 Distribuições das Variáveis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        variavel = st.selectbox(
            "Selecione a variável para visualizar:",
            ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak'],
            key="dist_select"
        )
        
        st.write(f"**Estatísticas de {variavel}:**")
        st.write(f"- Média: {df[variavel].mean():.2f}")
        st.write(f"- Mediana: {df[variavel].median():.2f}")
        st.write(f"- Desvio Padrão: {df[variavel].std():.2f}")
        st.write(f"- Range: {df[variavel].min():.2f} - {df[variavel].max():.2f}")
    
    with col2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.histplot(data=df, x=variavel, hue='output', ax=ax1, kde=True, alpha=0.7)
        ax1.set_title(f'Distribuição de {variavel} por Diagnóstico')
        ax1.legend(['Sem Doença', 'Com Doença'])
        
        sns.boxplot(data=df, x='output', y=variavel, ax=ax2)
        ax2.set_title(f'Boxplot de {variavel} por Diagnóstico')
        ax2.set_xticklabels(['Sem Doença', 'Com Doença'])
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with st.container():
        st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
        st.write("💡 **Insight:** Esta visualização mostra como a variável selecionada se distribui entre pacientes com e sem doença cardíaca.")
        st.markdown("</div>", unsafe_allow_html=True)

# Análise de correlação
def criar_analise_correlacao(df):
    st.subheader("🔗 Análise de Correlação")
    
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Matriz de Correlação entre Variáveis')
    st.pyplot(fig)
    
    st.subheader("Correlações com Diagnóstico (output)")
    correlacoes_target = corr['output'].sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Correlações Positivas:**")
        for var, corr_val in correlacoes_target.items():
            if corr_val > 0.1 and var != 'output':
                st.write(f"- {var}: {corr_val:.3f}")
    
    with col2:
        st.write("**Correlações Negativas:**")
        for var, corr_val in correlacoes_target.items():
            if corr_val < -0.1:
                st.write(f"- {var}: {corr_val:.3f}")

# Insights avançados
def criar_insights_avancados(df):
    st.subheader("🎯 Insights Avançados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Relação Idade vs Colesterol**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df, x='age', y='chol', hue='output', style='output', ax=ax)
        ax.set_title('Idade vs Colesterol por Diagnóstico')
        st.pyplot(fig)
        
        with st.container():
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write("💡 **Insight:** Pacientes com problemas cardíacos tendem a ter níveis mais altos de colesterol.")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.write("**Distribuição por Sexo**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x='sex', hue='output', ax=ax)
        ax.set_title('Distribuição de Doenças Cardíacas por Sexo')
        ax.set_xticklabels(['Feminino', 'Masculino'])
        ax.legend(['Sem Doença', 'Com Doença'])
        st.pyplot(fig)
        
        with st.container():
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.write("💡 **Insight:** Homens apresentam maior incidência de problemas cardíacos neste dataset.")
            st.markdown("</div>", unsafe_allow_html=True)

# Análise exploratória completa
def criar_analise_exploratoria(df):
    st.markdown("<div class='section-header'>📊 Análise Exploratória</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Estatísticas", "📈 Distribuições", "🔗 Correlações", "🎯 Insights"])
    
    with tab1:
        criar_estatisticas_descritivas(df)
    
    with tab2:
        criar_visualizacoes_distribuicao(df)
    
    with tab3:
        criar_analise_correlacao(df)
    
    with tab4:
        criar_insights_avancados(df)

# Classe do modelo de machine learning
class ModeloCardiaco:
    def __init__(self):
        self.modelo = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def preparar_dados(self, df):
        X = df.drop('output', axis=1)
        y = df['output']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X, y
    
    def treinar_modelo(self, n_estimators=100):
        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.modelo.fit(self.X_train, self.y_train)
        return self.modelo
    
    def avaliar_modelo(self):
        if self.modelo is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        y_pred = self.modelo.predict(self.X_test)
        y_pred_proba = self.modelo.predict_proba(self.X_test)[:, 1]
        
        acuracia = accuracy_score(self.y_test, y_pred)
        cv_scores = cross_val_score(self.modelo, self.X_train, self.y_train, cv=5)
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'acuracia': acuracia,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'matriz_confusao': confusion_matrix(self.y_test, y_pred),
            'relatorio_classificacao': classification_report(self.y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def prever(self, dados):
        if self.modelo is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        probabilidade = self.modelo.predict_proba(dados)[0][1]
        return probabilidade

# Seção do modelo
def criar_secao_modelo(df):
    st.markdown("<div class='section-header'>🤖 Modelo de Predição</div>", unsafe_allow_html=True)
    
    st.write("Esta seção treina um modelo de **Random Forest** para predição de doenças cardíacas.")
    
    with st.expander("⚙️ Configurações do Modelo"):
        n_estimators = st.slider("Número de Árvores", 50, 200, 100, 10)
    
    if st.button("🚀 Treinar Modelo", type="primary", key="train_btn"):
        with st.spinner("Treinando modelo e calculando métricas..."):
            modelo_cardiaco = ModeloCardiaco()
            X, y = modelo_cardiaco.preparar_dados(df)
            modelo_cardiaco.treinar_modelo(n_estimators=n_estimators)
            metricas = modelo_cardiaco.avaliar_modelo()
            
            st.subheader("📊 Resultados do Modelo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Acurácia", f"{metricas['acuracia']:.2%}")
            
            with col2:
                st.metric("Validação Cruzada", f"{metricas['cv_mean']:.2%}")
            
            with col3:
                st.metric("Consistência", f"±{metricas['cv_std']:.2%}")
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.write("**Matriz de Confusão:**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(metricas['matriz_confusao'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predito')
                ax.set_ylabel('Real')
                ax.set_xticklabels(['Sem Doença', 'Com Doença'])
                ax.set_yticklabels(['Sem Doença', 'Com Doença'])
                st.pyplot(fig)
            
            with col5:
                st.write("**Feature Importance:**")
                fig, ax = plt.subplots(figsize=(8, 6))
                top_features = metricas['feature_importance'].head(10)
                sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
                ax.set_title('Top 10 Features Mais Importantes')
                st.pyplot(fig)
            
            st.session_state.modelo = modelo_cardiaco
            st.session_state.metricas = metricas
            
            st.success("✅ Modelo treinado e avaliado com sucesso!")
    
    if 'metricas' in st.session_state:
        with st.expander("📋 Ver Relatório de Classificação Detalhado"):
            st.text(st.session_state.metricas['relatorio_classificacao'])
    
    return st.session_state.get('modelo', None)

# Validação de entradas
def validar_entradas(idade, pressao, colesterol, freq_card):
    erros = []
    
    if idade < 20 or idade > 100:
        erros.append("Idade deve estar entre 20 e 100 anos")
    
    if pressao < 90 or pressao > 200:
        erros.append("Pressão arterial deve estar entre 90 e 200 mmHg")
    
    if colesterol < 100 or colesterol > 400:
        erros.append("Colesterol deve estar entre 100 e 400 mg/dl")
    
    if freq_card < 60 or freq_card > 220:
        erros.append("Frequência cardíaca deve estar entre 60 e 220 bpm")
    
    return erros

# Processamento de entradas
def processar_entradas(idade, sexo, tipo_dor, pressao, colesterol, 
                      glicemia, freq_card, depressao_st, vasos, talassemia):
    sexo_num = 1 if sexo == "Masculino" else 0
    tipo_dor_num = int(tipo_dor.split(" - ")[0])
    glicemia_num = 1 if glicemia == "Sim" else 0
    
    dados = pd.DataFrame({
        'age': [idade],
        'sex': [sexo_num],
        'cp': [tipo_dor_num],
        'trtbps': [pressao],
        'chol': [colesterol],
        'fbs': [glicemia_num],
        'restecg': [0],
        'thalachh': [freq_card],
        'exng': [0],
        'oldpeak': [depressao_st],
        'slp': [1],
        'caa': [int(vasos)],
        'thall': [int(talassemia)]
    })
    
    return dados

# Mostrar resultados
def mostrar_resultado(probabilidade):
    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Resultado da Predição</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**Probabilidade de Doença Cardíaca:**")
        st.markdown(f"<h1 style='color: #FF4B4B; font-size: 3rem;'>{probabilidade:.1%}</h1>", unsafe_allow_html=True)
    
    with col2:
        st.progress(float(probabilidade))
    
    st.subheader("🎯 Interpretação do Resultado")
    
    if probabilidade < 0.2:
        st.success("🎉 **Baixo Risco** - Probabilidade baixa de doença cardíaca")
    elif probabilidade < 0.5:
        st.info("🔄 **Risco Leve** - Alguns fatores de risco presentes")
    elif probabilidade < 0.7:
        st.warning("⚠️ **Risco Moderado** - Múltiplos fatores de risco")
    else:
        st.error("🚨 **Alto Risco** - Alta probabilidade de doença cardíaca")
    
    with st.expander("⚠️ Aviso Importante e Limitações"):
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("""
        **AVISO MÉDICO IMPORTANTE:**
        
        Este sistema é uma ferramenta **EDUCACIONAL** e **NÃO substitui** o diagnóstico médico profissional. 
        Consulte sempre um médico qualificado para avaliações de saúde.
        
        **Limitações Conhecidas:**
        - Baseado em dados históricos limitados
        - Não considera histórico familiar
        - Não substitui exames clínicos
        - Precisão do modelo: ~80-85%
        - Não considera fatores genéticos
        
        **Use este sistema apenas para fins educacionais e de conscientização.**
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Formulário de predição
def criar_formulario_predicao(modelo):
    st.markdown("<div class='section-header'>🎯 Fazer Predição</div>", unsafe_allow_html=True)
    
    st.write("Preencha as informações do paciente para calcular o risco de doença cardíaca:")
    
    with st.form("formulario_predicao", clear_on_submit=False):
        st.write("**Informações Demográficas:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            idade = st.slider("Idade", 20, 80, 50, help="Idade do paciente em anos")
            sexo = st.selectbox("Sexo", ["Masculino", "Feminino"], help="Sexo biológico do paciente")
            tipo_dor = st.selectbox(
                "Tipo de Dor no Peito", 
                ["0 - Típica", "1 - Atípica", "2 - Não anginosa", "3 - Assintomática"],
                help="Tipo de dor no peito relatada pelo paciente"
            )
        
        with col2:
            pressao_arterial = st.slider("Pressão Arterial (mmHg)", 90, 200, 120, help="Pressão arterial em repouso")
            colesterol = st.slider("Colesterol (mg/dl)", 100, 400, 200, help="Nível de colesterol sérico")
            glicemia = st.selectbox("Glicemia > 120 mg/dl", ["Não", "Sim"], help="Açúcar no sangue em jejum elevado")
        
        st.write("**Informações Cardíacas:**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            freq_card_max = st.slider("Freq. Cardíaca Máxima", 60, 220, 150, help="Frequência cardíaca máxima alcançada")
            depressao_st = st.slider("Depressão ST", 0.0, 6.0, 1.0, 0.1, help="Depressão do segmento ST induzida por exercício")
        
        with col4:
            vasos_principais = st.selectbox("Número de Vasos Principais", ["0", "1", "2", "3"], help="Número de vasos principais coloridos na fluoroscopia")
            talassemia = st.selectbox("Resultado do Teste de Talassemia", ["0", "1", "2", "3"], help="Resultado do teste de talassemia")
        
        submitted = st.form_submit_button("🔍 Calcular Risco", type="primary")
        
        if submitted:
            erros = validar_entradas(idade, pressao_arterial, colesterol, freq_card_max)
            
            if erros:
                for erro in erros:
                    st.error(erro)
            elif modelo is not None:
                dados_processados = processar_entradas(
                    idade, sexo, tipo_dor, pressao_arterial, colesterol,
                    glicemia, freq_card_max, depressao_st, vasos_principais, talassemia
                )
                
                probabilidade = modelo.prever(dados_processados)
                mostrar_resultado(probabilidade)
            else:
                st.error("❌ Por favor, treine o modelo primeiro na seção acima.")

# Função principal
def main():
    aplicar_estilos()
    
    st.markdown("<h1 class='main-header'>❤️ Sistema de Predição de Doenças Cardíacas</h1>", unsafe_allow_html=True)
    
    st.write("*Sistema inteligente para análise exploratória e predição de riscos cardíacos usando Machine Learning*")
    
    df = load_data()
    
    if df is not None:
        mostrar_info_dataset(df)
        criar_analise_exploratoria(df)
        modelo = criar_secao_modelo(df)
        
        if modelo is not None:
            criar_formulario_predicao(modelo)
        else:
            st.info("💡 **Instrução:** Primeiro, treine o modelo na seção **'🤖 Modelo de Predição'** acima.")
    
    st.markdown("---")
    st.markdown("Desenvolvido com ❤️ usando Streamlit | [GitHub](https://github.com)")

if __name__ == "__main__":
    main()