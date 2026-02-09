import streamlit as st
import pandas as pd
import os
import numpy as np
import google.generativeai as genai
from datetime import datetime
from sentence_transformers import SentenceTransformer
defaults = {
    "api_key": None,
    "llm": None,
    "model": None,
    "df": None,
    "texts": None,
    "embeddings": None,
    "row_indices": None,
    "messages": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
def row_to_text(row,df):
        return " | ".join([f"{col}: {row[col]}" for col in df.columns])

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
@st.cache_data
def process_files(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    texts = [row_to_text(row,df) for _, row in df.iterrows()]
    row_indices = list(df.index)

    if uploaded_file:
        st.session_state.messages = []

    model = load_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return df,texts,embeddings,row_indices 

    
st.title("RAG")  

    
uploaded_file = st.file_uploader("Choose file", type=['csv'])

if uploaded_file:  
    df,texts,embeddings,row_indices =process_files(uploaded_file)
    st.session_state.model = load_model()
    st.session_state.df = df
    st.session_state.texts = texts
    st.session_state.embeddings = embeddings
    st.session_state.row_indices=row_indices
    name=uploaded_file.name
    st.write("File: ",name)
    st.write("rows: ",df.shape[0])  
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(5))
    
b=st.sidebar
b.title("API Key")
api_key=b.text_input(type="password",label="Enter API Key")
if api_key and st.session_state.llm is None:
    st.session_state.api_key = api_key
    b.write(api_key[:3]+"xxxxxxxxxxxx")
    genai.configure(api_key=st.session_state.api_key)
    st.session_state.llm = genai.GenerativeModel("gemini-2.5-flash")
    st.sidebar.success("LLM Ready")

ready = True

if st.session_state.llm is None:
    st.info("Enter API key in the sidebar")
    ready = False

if st.session_state.embeddings is None:
    st.info("Upload a CSV dataset")
    ready = False



#function to get response form LLM
def get_resp(prompt):
    llm = st.session_state.llm
    response = llm.generate_content(prompt)
    try:
        return response.text
    except AttributeError:
       
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        return f"Error: {str(e)}"

#Reterieve rows most related to the question
def retrieve_rows(query: str, top_k: int = 5):
    sbert_model = st.session_state.model
    embeddings = st.session_state.embeddings
    texts = st.session_state.texts
    row_indices= st.session_state.row_indices
    df = st.session_state.df
    q_emb = sbert_model.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / np.linalg.norm(q_emb)

    scores = embeddings.dot(q_emb)  # cosine scores (since normalized)
    N = scores.shape[0]

    if top_k >= N:
        top_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        row_idx = row_indices[idx]
        row_text = texts[idx]
        score = float(scores[idx])
        results.append((row_idx, row_text, score))
    return results

#generating answers for questions using rag
def rag_answer(query):
    retrieved_rows = retrieve_rows(query)

    #for idx, text in retrieved_rows:
     #   print(f"Respondent Index: {idx}")
      #  print(text)
       # print("-" * 50)

    context_block = "\n\n".join([text for _, text, _ in retrieved_rows])

    prompt = f"""
    You should ack as a data analyst. Use ONLY the following rows from the dataset.

    Dataset Rows:
    {context_block}

    Question:
    {query}

    Now answer clearly,
    and show any calculations you perform.
    """

    answer = get_resp(prompt)
    return answer

if "messages" not in st.session_state:
    st.session_state.messages = []
    

    

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)

# ---------------- CHAT INPUT ----------------
q = st.chat_input("Ask a question about the dataset...", disabled=not ready)

if q:
    # Save user message
    st.session_state.messages.append(("user", q))
    # Generate answer
    with st.spinner("Thinking..."):
        answer = rag_answer(q)
    # Save assistant message
    st.session_state.messages.append(("assistant", answer))
    # Force rerun so messages show once
    st.rerun()
