import streamlit as st
import pandas as pd
import os
import numpy as np
import google.generativeai as genai
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os
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
def process_pdf(uploaded_file):
    """Extract and chunk text from PDF"""
    
    # Save uploaded file to temporary location
    # (LangChain's PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(pages)
        
        # Extract text from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        return texts, len(pages)  # Return texts and page count
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
@st.cache_data
def process_files(uploaded_file):
    name=uploaded_file.name
    if name.endswith(".pdf"):
        texts, page_count = process_pdf(uploaded_file)
        metadata = {'type': 'pdf', 'pages': page_count}
        df=None
        row_indices=None
    elif name.endswith(".csv"):
        
        df = pd.read_csv(uploaded_file)
        
        texts = [row_to_text(row,df) for _, row in df.iterrows()]
        metadata = {'type': 'csv', 'rows': df.shape[0]}
        row_indices = list(df.index)
    else:
        raise ValueError("Unsupported file type")
    if uploaded_file:
        st.session_state.messages = []

    
    model = load_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return df,texts,embeddings,row_indices,metadata 

    
st.title("RAG")  

    
uploaded_file = st.file_uploader("Choose file", type=['pdf','csv'])

if uploaded_file:  
    df,texts,embeddings,row_indices,metadata =process_files(uploaded_file)
    st.session_state.model = load_model()
    st.session_state.df = df
    st.session_state.texts = texts
    st.session_state.embeddings = embeddings
    st.session_state.row_indices=row_indices
    if metadata['type'] == 'pdf':
        col1, col2, col3 = st.columns(3)
        col1.metric("File", uploaded_file.name)
        col2.metric("Pages", metadata['pages'])
        col3.metric("Chunks", len(texts))
        
    elif metadata['type'] == 'csv':
        col1, col2, col3 = st.columns(3)
        col1.metric("File", uploaded_file.name)
        col2.metric("Rows", metadata['rows'])
        col3.metric("Chunks", len(texts))
    
    if metadata['type'] == 'csv' and 'df' in st.session_state:
        with st.expander("Preview Dataset"):
            st.dataframe(st.session_state.df.head(10))
    
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
def retrieve_rows(query: str, top_k: int = 10):
    sbert_model = st.session_state.model
    embeddings = st.session_state.embeddings
    texts = st.session_state.texts
    row_indices= st.session_state.row_indices
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
        row_idx = row_indices[idx] if row_indices is not None else None
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

    avg_score = sum(score for _, _, score in retrieved_rows) / len(retrieved_rows)
    
    if avg_score > 0.7:
        st.success(f"High confidence ({avg_score:.0%})")
    elif avg_score > 0.4:
        st.warning(f"Medium confidence ({avg_score:.0%})")
    else:
        st.error(f"Low confidence ({avg_score:.0%}) - answer may not be accurate")
    

    prompt = f"""
    You are a helpful assistant. Use ONLY the following information to answer the question.

    Dataset Rows:
    {context_block}

    Question:
    {query}

    Now answer clearly,
    and If the answer is not in the context, say so.
    """

    answer = get_resp(prompt)
    with st.expander("View Sources Used"):
        for i, (idx, text, score) in enumerate(retrieved_rows):
            st.write(f"**Chunk {i+1}** (relevance: {score:.2f})")
            st.caption(text[:300] + "...")
            st.divider()
    return answer,avg_score, retrieved_rows

if "messages" not in st.session_state:
    st.session_state.messages = []
    

    
if st.session_state.messages:
    # Build text content
    chat_text = "\n\n".join([
        f"{role.upper()}: {msg}" 
        for role, msg,meta in st.session_state.messages
    ])
    
    st.download_button(
        label="Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

for role, msg, meta in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)
        
        if role == "assistant" and meta:
            score = meta["score"]
            chunks = meta["chunks"]
            
            # Confidence score
            if score > 0.5:
                st.success(f"High confidence ({score:.0%})")
            elif score > 0.3:
                st.warning(f"Medium confidence ({score:.0%})")
            else:
                st.error(f"Low confidence ({score:.0%})")
            
            # Retrieved chunks
            with st.expander(f"Sources ({len(chunks)} chunks)"):
                for i, (idx, text, chunk_score) in enumerate(chunks):
                    st.write(f"**Chunk {i+1}** (relevance: {chunk_score:.2f})")
                    preview = text[:300] + "..." if len(text) > 300 else text
                    st.caption(preview)
                    st.divider()

# ---------------- CHAT INPUT ----------------
q = st.chat_input("Ask a question about the dataset...", disabled=not ready)

if q:
    # Save user message
    st.session_state.messages.append(("user", q,None))
    # Generate answer
    with st.spinner("Thinking..."):
        answer,avg_score, chunks = rag_answer(q)
    # Save assistant message
    st.session_state.messages.append((
        "assistant", 
        answer, 
        {"score": avg_score, "chunks": chunks}
    ))
    
    # Force rerun so messages show once
    st.rerun()
