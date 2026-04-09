import pandas as pd
from utils import row_to_text, safe_exec_pandas, parse_json, format_history

def test_row_to_text():
    df = pd.DataFrame([{"name": "John", "age": 30}])
    row = df.iloc[0]
    result = row_to_text(row, df)
    assert "name: John" in result
    assert "age: 30" in result

def test_safe_exec_pandas_valid():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = safe_exec_pandas("df['x'].sum()", df)
    assert result == 6

def test_safe_exec_pandas_blocked():
    df = pd.DataFrame({"x": [1, 2]})
    result = safe_exec_pandas("import os", df)
    assert result == "Unsafe code detected."

def test_parse_json_valid():
    text = 'some text {"action": "search", "query": "abc"} more text'
    result = parse_json(text)
    assert result["action"] == "search"

def test_parse_json_invalid():
    result = parse_json("garbage response")
    assert result["action"] == "answer"

def test_format_history():
    history = [
        {
            "action": "search",
            "query": "test",
            "thought": "need info",
            "results": [("id", "text", 0.9)]
        }
    ]
    output = format_history(history)
    assert "Agent searched for: test" in output