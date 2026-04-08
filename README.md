# RAG-Powered Document Q&A System

A hybrid RAG + agentic data analysis system that combines semantic retrieval with dynamic computation over structured data. It is built with Streamlit that enables intelligent question-answering over PDF documents and CSV datasets using Google's Gemini LLM and semantic search. 

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Multi-format Support**: Process both PDF documents and CSV datasets
- **Semantic Search**: Uses sentence transformers for accurate context retrieval
- **RAG Pipeline**: Combines retrieval with Google Gemini 2.0 Flash for accurate answers
- **Confidence Scoring**: Displays relevance scores for transparency
- **Source Attribution**: Shows which document chunks were used for each answer
- **Chat History**: Maintains conversation context with downloadable chat logs
- **Robust Error Handling**: Production-ready error management and logging
- **Interactive UI**: Clean Streamlit interface with real-time processing
- **Agentic RAG Pipeline**: Dynamically decides whether to retrieve context (SEARCH) or perform data computations (COMPUTE)
- **Natural Language Data Analysis**: Converts user queries into executable pandas operations for CSV datasets
- **Hybrid Reasoning Engine**: Combines semantic retrieval (PDF) with structured computation (CSV)
- **Safe Code Execution**: Executes LLM-generated pandas code in a sandboxed environment
- **Multi-step Query Planning**: Iteratively refines search and computation steps before answering
  
## 🏗️ Architecture

```
User Query
    ↓
Query Embedding (SentenceTransformer)
    ↓
Agent Reasoning (LLM decides next action)  
    ↓  
 ┌───────────────┬───────────────┐  
 │ SEARCH        │ COMPUTE        │  
 │ (PDF/CSV RAG) │ (CSV only)     │  
 │               │               │  
 │ Embeddings    │ Pandas Code    │  
 │ + Retrieval   │ Generation     │  
 │               │ + Safe Exec    │  
 └───────────────┴───────────────┘  
    ↓  
Context Aggregation  
    ↓  
Final Answer Generation (Gemini)  
    ↓
Answer + Sources
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.0 Flash (via google-generativeai)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Document Processing**: LangChain (PyPDFLoader, RecursiveCharacterTextSplitter)
- **Data Processing**: Pandas, NumPy
- **Language**: Python 3.8+
- **Agent Framework**: Custom iterative reasoning agent (LLM-driven decision making)
- **Execution Engine**: Safe pandas evaluation layer for dynamic query execution

## 🧠 Intelligent Query Handling

The system dynamically adapts to different query types:

- **Qualitative Questions (PDF/CSV)**  
  → Uses semantic search + RAG  
  _Example: "What are the main risks mentioned in the document?"_

- **Quantitative Questions (CSV only)**  
  → Converts natural language into pandas operations  
  _Example: "What is the average salary of engineers?"_

- **Hybrid Reasoning**  
  → Combines retrieval and computation when needed  

This hybrid approach enables both **contextual understanding** and **data-driven insights** within a single interface.

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key 


### Run the application
```
streamlit run app.py
```

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
langchain>=0.1.0
langchain-community>=0.0.10
pypdf>=3.17.0
```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Enter API Key**
   - Navigate to the sidebar
   - Enter your Google Gemini API key
   - Key is masked for security

3. **Upload Document**
   - Support formats: PDF, CSV
   - Maximum file size: 50MB
   - The app will automatically process and chunk the document

4. **Ask Questions**
   - Type your question in the chat input
   - Receive AI-generated answers with source attribution
   - View confidence scores and relevant document chunks

5. **Download Chat History**
   - Click "Download Chat" button to export conversation
## 🚀 Advanced Capabilities

- Agentic reasoning loop with iterative decision-making
- Dynamic query routing (SEARCH vs COMPUTE)
- Natural language to code translation (pandas)
- Confidence scoring based on embedding similarity
- Explainability via retrieved source chunks

## 💡 Use Cases

- **Document Analysis**: Query technical documentation, research papers, or reports
- **Dataset Exploration**: Ask natural language questions about CSV data
- **Knowledge Management**: Build searchable knowledge bases from PDFs
- **Research Assistant**: Quick information retrieval from large documents
- **Data Analytics**: Conversational interface for exploring tabular data
- **AI Data Assistant**: Ask complex analytical questions over structured datasets
- **Hybrid Knowledge Systems**: Combine unstructured (PDF) and structured (CSV) data
- **Conversational Analytics**: Replace dashboards with natural language queries
  
## 🔧 Configuration

### Embedding Model
The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. To change:

```python
@st.cache_resource
def load_model():
    return SentenceTransformer("your-model-name")
```

### Chunking Parameters
Adjust text splitting in `process_pdf()`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust chunk size
    chunk_overlap=200,    # Adjust overlap
    length_function=len,
)
```

### Retrieval Settings
Modify top-k results in `retrieve_rows()`:

```python
def retrieve_rows(query: str, top_k: int = 10):  # Change top_k
```

## 📊 Performance

- **Embedding Generation**: ~100 chunks/second
- **Query Response Time**: 2-5 seconds (depends on LLM API)
- **Memory Usage**: ~500MB for model + document size
- **Concurrent Users**: Supports Streamlit's default session management

## 🛡️ Error Handling

The application includes comprehensive error handling for:

- ✅ File upload validation (size, format, corruption)
- ✅ PDF parsing errors
- ✅ CSV parsing errors
- ✅ API key validation
- ✅ Rate limiting and retries
- ✅ Embedding generation failures
- ✅ Query processing errors
- ✅ LLM timeout and safety filters


## 🔒 Security Considerations

- API keys are stored in session state (not persisted)
- Input validation prevents injection attacks
- File size limits prevent memory exhaustion
- Logging excludes sensitive information
- Sandboxed execution of LLM-generated code (restricted eval environment)
- Prevention of arbitrary code execution via keyword filtering and scoped evaluation
- Separation of retrieval and computation pipelines to reduce attack surface
  
## 📈 Future Enhancements

- [ ] Multi-document support (upload multiple files)
- [ ] Vector database integration (Pinecone, Chroma)
- [ ] Advanced filtering and metadata search
- [ ] Export answers to PDF/Word
- [ ] User authentication and session management
- [ ] Cost tracking for API usage
- [ ] Fine-tuned embeddings for domain-specific use cases
- [ ] Support for additional file formats (DOCX, TXT, HTML)

## 🙏 Acknowledgments

- Google Gemini for the LLM API
- Sentence Transformers for embedding models
- LangChain for document processing utilities
- Streamlit for the interactive framework


---

**Note**: This project requires a Google Gemini API key. Free tier includes generous limits for development and testing.
