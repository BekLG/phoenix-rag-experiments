import sys

def test_evaluation_module_imports_cleanly():
    """Test that evaluator module imports cleanly without missing mistral dependencies."""
    # Remove it if already imported to simulate a fresh import
    if "phoenix_rag.evaluation.evaluator" in sys.modules:
        del sys.modules["phoenix_rag.evaluation.evaluator"]
    
    import phoenix_rag.evaluation.evaluator
    assert phoenix_rag.evaluation.evaluator is not None
