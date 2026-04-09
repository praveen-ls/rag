import re
import json
import pandas as pd

def row_to_text(row,df):
        return " | ".join([f"{col}: {row[col]}" for col in df.columns])
def safe_exec_pandas(code: str, df: pd.DataFrame):
    # Allowed variables
    allowed_globals = {
        "__builtins__": {},  # block builtins
    }

    allowed_locals = {
        "df": df,
        "pd": pd
    }

    # Block dangerous patterns
    forbidden = ["import", "__", "os", "sys", "eval", "exec", "open", "write", "read"]

    if any(word in code.lower() for word in forbidden):
        return "Unsafe code detected."

    try:
        result = eval(code, allowed_globals, allowed_locals)
        return result
    except Exception as e:
        return f"Execution error: {str(e)}"
def parse_json(response_text):
    """Extract JSON from LLM response and convert to dict"""
    print(response_text)
    try:
        # Sometimes LLM wraps JSON in ```json ```
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except Exception:
        pass

    # fallback if model responds badly
    return {
        "action": "answer",
        "reasoning": "Could not parse model output",
        "query": ""
    }
def format_history(history):
    """Convert agent history into readable text for the LLM"""
    
    if not history:
        return "No previous searches."

    formatted = []
    for step in history:
        if step["action"] == "search":
            formatted.append(
                f"Agent searched for: {step['query']}\n"
                f"Reason: {step['thought']}\n"
                f"Found {len(step['results'])} results."
            )

    return "\n\n".join(formatted)

