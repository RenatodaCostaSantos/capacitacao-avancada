import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os


import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

st.title("📊 Análise Inteligente de CSV com PandasAI")

uploaded_file = st.file_uploader("Upload do CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview")
    st.dataframe(df.head())

    # Configurar LLM
    llm = OpenAI(api_token=OPENAI_API_KEY)

    # Criar SmartDataframe
    sdf = SmartDataframe(df, config={"llm": llm})

    


    pergunta = st.text_input("Faça uma pergunta sobre os dados")

    if pergunta:
        #resposta = sdf.chat(pergunta)
        resposta = sdf.chat(f"Responda em português do Brasil. {pergunta}")
        st.write("### 🤖 Resposta")
        st.write(resposta)
