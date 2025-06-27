'''
Esse é um protótipo inicial de uma aplicação web utilizando Streamlit para recomendar trilhas de capacitação industrial com base em um questionário.
O código utiliza um modelo de machine learning (RandomForestClassifier) treinado com dados fictícios de perfis profissionais e trilhas de capacitação do SESI.

Necessário baixar os seguintes módulos:
- streamlit
- pandas
- json
- os
- matplotlib (opcional, para visualização)
- seaborn (opcional, para visualização)
- numpy (opcional, para manipulação de arrays)
- joblib (opcional, para salvar o modelo treinado)
- scikit-learn (para o modelo de machine learning)


Atenção: Para executar este código você deverá instalar os módulos acima e executar o comando `streamlit run app.py` no terminal, na pasta onde o arquivo `app.py` está localizado.
O arquivo `perfis.json` e `opcoes.json` devem estar na pasta `data` dentro do diretório onde o `app.py` está localizado.
'''

import streamlit as st
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def carregar_json(caminho):
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)

caminho_perfis = os.path.join("data", "perfis.json")
caminho_opcoes = os.path.join("data", "opcoes.json")

perfis = carregar_json(caminho_perfis)
opcoes = carregar_json(caminho_opcoes)


df = pd.DataFrame(perfis)

campos_input = ['funcao', 'experiencia', 'escolaridade', 'interesse', 'tecnologia', 'automacao', 'metodologia', 'turno']
X = df[campos_input]
y = df['trilha']


encoders = {}
X_encoded = pd.DataFrame()
for col in campos_input:
    enc = LabelEncoder()
    todas_opcoes = opcoes[col]
    enc.fit(todas_opcoes) 
    X_encoded[col] = enc.transform(X[col])
    encoders[col] = enc



y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)


modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_encoded, y_encoded)


st.set_page_config(page_title="Rede Capacita IA", page_icon="🤖")
st.title("Questionário Inteligente")
st.markdown("Sistema inteligente para recomendação de **trilhas de capacitação industrial** SESI")

with st.form("quiz_form"):
    st.subheader("Perfil Profissional")
    funcao = st.selectbox("1. Qual sua função atual?", opcoes['funcao'])
    experiencia = st.selectbox("2. Quanto tempo de experiência você tem na indústria?", opcoes['experiencia'])
    escolaridade = st.selectbox("3. Qual seu grau de escolaridade?", opcoes['escolaridade'])

    st.subheader("Interesses e Habilidades")
    interesse = st.selectbox("4. O que você gostaria de aprender?", opcoes['interesse'])
    tecnologia = st.selectbox("5. Você se sente confortável com tecnologia no dia a dia?", opcoes['tecnologia'])
    automacao = st.selectbox("6. Você já utilizou algum sistema automatizado no seu trabalho?", opcoes['automacao'])

    st.subheader("Preferências de Aprendizado")
    metodologia = st.selectbox("7. Prefere cursos com mais prática ou mais teoria?", opcoes['metodologia'])
    turno = st.selectbox("8. Qual turno você estaria disponível para estudar?", opcoes['turno'])

    submitted = st.form_submit_button("🔍 Recomendar Trilha")


if submitted:
    entrada_usuario = {
        "funcao": funcao,
        "experiencia": experiencia,
        "escolaridade": escolaridade,
        "interesse": interesse,
        "tecnologia": tecnologia,
        "automacao": automacao,
        "metodologia": metodologia,
        "turno": turno
    }

    entrada_codificada = []
    for campo in campos_input:
        valor = entrada_usuario[campo]
        valor_cod = encoders[campo].transform([valor])[0]
        entrada_codificada.append(valor_cod)


    trilha_prevista_cod = modelo.predict([entrada_codificada])[0]
    trilha_prevista = y_encoder.inverse_transform([trilha_prevista_cod])[0]


    perfil_encontrado = next((p for p in perfis if p['trilha'] == trilha_prevista), None)

    st.success("### Resultado sugerido (via IA):")
    st.markdown(f"**Área do conhecimento recomendada:** {trilha_prevista}")

    if perfil_encontrado:
        st.markdown("**Trilha SESI sugerida:**")
        for curso in perfil_encontrado['cursos']:
            st.markdown(f"- {curso}")
        st.info(f"**Meta sugerida:** {perfil_encontrado['meta']}")
    else:
        st.warning("Não encontramos cursos específicos para essa trilha.")

st.caption("Protótipo com IA | Recomendação via RandomForestClassifier | Modelo SESI de capacitação industrial")
