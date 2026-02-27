"""
MedAgent 360 · Phase 0 Smoke Tests
Run this after setup to verify all tools are working.
Usage: python scripts/smoke_test.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []

def check(name, fn):
    try:
        fn()
        RESULTS.append(("✅", name))
        print(f"✅  {name}")
    except Exception as e:
        RESULTS.append(("❌", f"{name} — {e}"))
        print(f"❌  {name} — {e}")


# ── Dependency Checks ─────────────────────────────────────────────────────────

def check_pdfplumber():
    import pdfplumber
    assert pdfplumber.__version__

def check_langchain():
    import langchain
    assert langchain.__version__

def check_chromadb():
    import chromadb
    client = chromadb.EphemeralClient()
    col = client.create_collection("smoke_test")
    col.add(documents=["test"], ids=["1"])
    assert col.count() == 1

def check_gtts():
    from gtts import gTTS
    tts = gTTS(text="test", lang="en")
    assert tts

def check_gemini_key():
    from scripts.config import config
    missing = config.validate()
    if missing:
        raise ValueError(f"Missing: {missing}")

def check_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(["test"])
    assert emb.shape[0] == 1

def check_fastapi():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200

def check_streamlit():
    import streamlit
    assert streamlit.__version__

def check_module_a_imports():
    from lab_report.pdf_parser import extract_lab_values
    from lab_report.vector_store import build_vector_store, query_benchmark
    from lab_report.rag_pipeline import classify_lab_values, generate_summary
    from lab_report.voice import generate_audio


# ── Run All Checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MedAgent 360 · Phase 0 Smoke Tests")
    print("="*55 + "\n")

    check("pdfplumber import", check_pdfplumber)
    check("langchain import", check_langchain)
    check("chromadb (ephemeral)", check_chromadb)
    check("gTTS import", check_gtts)
    check("Gemini API key set", check_gemini_key)
    check("sentence-transformers", check_sentence_transformers)
    check("FastAPI /health endpoint", check_fastapi)
    check("streamlit import", check_streamlit)
    check("Module A imports", check_module_a_imports)

    print("\n" + "="*55)
    passed = sum(1 for r in RESULTS if r[0] == "✅")
    failed = sum(1 for r in RESULTS if r[0] == "❌")
    print(f"  Results: {passed} passed / {failed} failed")
    print("="*55 + "\n")

    if failed > 0:
        print("Fix the failing checks before Phase 1 development.\n")
        sys.exit(1)
    else:
        print("🚀 All checks passed! Ready to build.\n")
