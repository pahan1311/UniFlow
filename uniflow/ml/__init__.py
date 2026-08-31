"""Model loading, inference, explainability and embeddings.

This is the only package that imports torch or transformers. Keeping it
isolated is what lets the Streamlit client and the test suite run without
the ~1 GB of weights the old design loaded in every process.
"""
