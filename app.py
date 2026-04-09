import streamlit as st
from rag_core import process_files, load_model, agentic_rag
from google import genai

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
    
st.title("RAG")  

    
uploaded_file = st.file_uploader("Choose file", type=['pdf','csv'])

if uploaded_file:  
    df,texts,embeddings,row_indices,metadata =process_files(uploaded_file)
    st.session_state.model = load_model()
    st.session_state.df = df
    st.session_state.texts = texts
    st.session_state.embeddings = embeddings
    st.session_state.row_indices=row_indices
    st.session_state.data_type = metadata['type']
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
    #genai.configure(api_key=st.session_state.api_key)
    st.session_state.llm = genai.Client(api_key=st.session_state.api_key)
    #st.session_state.llm = genai.GenerativeModel(
    #model="gemini-2.5-flash",
    #api_key=st.session_state.api_key
    #)
    #st.session_state.llm = genai.GenerativeModel("gemini-2.5-flash")
    st.sidebar.success("LLM Ready")

ready = True

if st.session_state.llm is None:
    st.info("Enter API key in the sidebar")
    ready = False

if st.session_state.embeddings is None:
    st.info("Upload a CSV dataset")
    ready = False

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
st.session_state.is_thinking = False
q = st.chat_input("Ask a question about the dataset...", disabled=not ready)

if q:
    # Save user message
    st.session_state.messages.append(("user", q,None))
    st.session_state.is_thinking = True
    # Generate answer
    with st.spinner("Thinking..."):
        answer = agentic_rag(q)
        avg_score = 1.0
        chunks = []
    # Save assistant message
    st.session_state.messages.append((
        "assistant", 
        answer, 
        {"score": avg_score, "chunks": chunks}
    ))
    st.session_state.is_thinking = False
    # Force rerun so messages show once
    st.rerun()
