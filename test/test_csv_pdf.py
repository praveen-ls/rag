import pandas as pd
from io import StringIO
from rag_core import process_files

class FakeModel:
    def encode(self, texts, **kwargs):
        import numpy as np
        return np.ones((len(texts), 2))

def test_process_csv(monkeypatch):
    csv_data = "name,age\nJohn,30\nJane,25"
    file = StringIO(csv_data)
    file.name = "test.csv"

    monkeypatch.setattr("rag_core.load_model", lambda: FakeModel())

    df, texts, embeddings, row_indices, metadata = process_files(file)

    assert metadata["type"] == "csv"
    assert len(texts) == 2
    assert df.shape[0] == 2
from io import BytesIO

def test_process_pdf(monkeypatch):
    fake_pdf = BytesIO(b"%PDF-1.4\n%EOF")
    fake_pdf.name = "test.pdf"

    # mock PDF loader
    monkeypatch.setattr("rag_core.process_pdf", lambda f: (["text1", "text2"], 2))
    monkeypatch.setattr("rag_core.load_model", lambda: FakeModel())

    df, texts, embeddings, row_indices, metadata = process_files(fake_pdf)

    assert metadata["type"] == "pdf"
    assert len(texts) == 2