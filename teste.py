import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os

st.set_page_config(page_title="CSV + IA Analyzer", layout="wide")

st.title("📊 Analisador Inteligente de CSV com IA")

uploaded_file = st.file_uploader("Faça upload do seu CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview dos dados")
    st.dataframe(df.head())

    # Converter linhas em texto
    textos = df.astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()

    # Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(textos)

    # Criar índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    pergunta = st.text_input("Faça uma pergunta sobre os dados")

    if pergunta:
        query_embedding = model.encode([pergunta])
        D, I = index.search(np.array(query_embedding), k=5)

        contexto = "\n".join([textos[i] for i in I[0]])

        st.write("### Contexto recuperado:")
        st.write(contexto)

        # Enviar para LLM
        openai.api_key = os.getenv("OPENAI_API_KEY")

        resposta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Você é um analista de dados."},
                {"role": "user", "content": f"Contexto:\n{contexto}\n\nPergunta:{pergunta}"}
            ]
        )

        st.write("### 🤖 Resposta da IA")
        st.write(resposta["choices"][0]["message"]["content"])
