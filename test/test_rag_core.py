import numpy as np
import streamlit as st
from rag_core import retrieve_rows, agentic_rag

class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts])

class FakeLLM:
    def __init__(self):
        self.calls = 0

    def models(self):
        return self

    def generate_content(self, model=None, contents=None):
        class Resp:
            text = '{"action": "answer", "query": ""}'
        return Resp()
def test_retrieve_rows():
    st.session_state.model = FakeModel()
    st.session_state.texts = ["hello world", "goodbye world"]
    st.session_state.embeddings = np.array([[1, 0], [0, 1]])
    st.session_state.row_indices = [0, 1]

    results = retrieve_rows("hello", top_k=1)

    assert len(results) == 1
    assert "hello world" in results[0][1]
def test_agentic_rag_basic(monkeypatch):
    st.session_state.model = FakeModel()
    st.session_state.embeddings = np.array([[1, 0]])
    st.session_state.texts = ["sample text"]
    st.session_state.row_indices = [0]
    st.session_state.data_type = "pdf"

    # mock agent_think → always answer
    monkeypatch.setattr(
        "rag_core.agent_think",
        lambda q, h: {"action": "answer", "reasoning": "", "query": ""}
    )

    # mock LLM response
    monkeypatch.setattr(
        "rag_core.get_resp",
        lambda prompt: "final answer"
    )

    result = agentic_rag("question")
    assert result == "final answer"