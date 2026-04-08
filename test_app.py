def test_app_runs():
    try:
        import app
    except Exception as e:
        assert False, f"App failed to start: {e}"
