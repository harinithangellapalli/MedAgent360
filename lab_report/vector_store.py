"""
MedAgent 360 · Module A · Step A2
ChromaDB Vector Store Builder
Builds and queries the medical benchmark RAG database.
Normal ranges sourced from standard clinical references.
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("lab_report.vector_store")

# ── Embedded Medical Benchmark Data ───────────────────────────────────────────
# Format: test_name → {min, max, unit, description}
MEDICAL_BENCHMARKS = [
    # Complete Blood Count (CBC)
    {"test": "Hemoglobin", "min": 13.0, "max": 17.0, "unit": "g/dL", "gender": "male", "description": "Oxygen-carrying protein in red blood cells. Low values indicate anemia."},
    {"test": "Hemoglobin", "min": 12.0, "max": 15.5, "unit": "g/dL", "gender": "female", "description": "Oxygen-carrying protein in red blood cells. Low values indicate anemia."},
    {"test": "WBC", "min": 4.5, "max": 11.0, "unit": "10³/µL", "gender": "all", "description": "White blood cells fight infection. High values may indicate infection or inflammation."},
    {"test": "Platelets", "min": 150.0, "max": 400.0, "unit": "10³/µL", "gender": "all", "description": "Clotting cells. Low values increase bleeding risk; high values may indicate clotting risk."},
    {"test": "RBC", "min": 4.5, "max": 5.5, "unit": "10⁶/µL", "gender": "male", "description": "Red blood cells carry oxygen throughout the body."},
    {"test": "RBC", "min": 4.0, "max": 5.0, "unit": "10⁶/µL", "gender": "female", "description": "Red blood cells carry oxygen throughout the body."},
    {"test": "Hematocrit", "min": 40.0, "max": 52.0, "unit": "%", "gender": "male", "description": "Percentage of blood volume made up of red blood cells."},
    {"test": "Hematocrit", "min": 36.0, "max": 48.0, "unit": "%", "gender": "female", "description": "Percentage of blood volume made up of red blood cells."},
    {"test": "MCV", "min": 80.0, "max": 100.0, "unit": "fL", "gender": "all", "description": "Mean corpuscular volume — average size of red blood cells."},
    # Liver Function
    {"test": "SGPT", "min": 7.0, "max": 40.0, "unit": "U/L", "gender": "all", "description": "ALT liver enzyme. Elevated levels indicate liver cell damage."},
    {"test": "ALT", "min": 7.0, "max": 40.0, "unit": "U/L", "gender": "all", "description": "Alanine aminotransferase — liver enzyme. High values suggest liver damage."},
    {"test": "SGOT", "min": 10.0, "max": 40.0, "unit": "U/L", "gender": "all", "description": "AST enzyme found in liver and heart. Elevated in liver disease or heart attack."},
    {"test": "AST", "min": 10.0, "max": 40.0, "unit": "U/L", "gender": "all", "description": "Aspartate aminotransferase. Elevated in liver disease or heart attack."},
    {"test": "Bilirubin Total", "min": 0.2, "max": 1.2, "unit": "mg/dL", "gender": "all", "description": "Waste product from red blood cell breakdown. High levels cause jaundice."},
    # Kidney Function
    {"test": "Creatinine", "min": 0.7, "max": 1.3, "unit": "mg/dL", "gender": "male", "description": "Kidney waste marker. Elevated values indicate reduced kidney function."},
    {"test": "Creatinine", "min": 0.5, "max": 1.1, "unit": "mg/dL", "gender": "female", "description": "Kidney waste marker. Elevated values indicate reduced kidney function."},
    {"test": "Blood Urea Nitrogen", "min": 7.0, "max": 20.0, "unit": "mg/dL", "gender": "all", "description": "BUN — kidney function marker. High values may indicate kidney disease or dehydration."},
    {"test": "BUN", "min": 7.0, "max": 20.0, "unit": "mg/dL", "gender": "all", "description": "Blood urea nitrogen — kidney function indicator."},
    {"test": "Uric Acid", "min": 3.5, "max": 7.2, "unit": "mg/dL", "gender": "male", "description": "High levels can cause gout. Low levels are rare but may indicate kidney issues."},
    # Diabetes / Blood Sugar
    {"test": "Fasting Blood Glucose", "min": 70.0, "max": 100.0, "unit": "mg/dL", "gender": "all", "description": "Blood sugar after fasting. 100–125 is pre-diabetic; above 126 is diabetic range."},
    {"test": "Blood Glucose Fasting", "min": 70.0, "max": 100.0, "unit": "mg/dL", "gender": "all", "description": "Fasting blood sugar level."},
    {"test": "HbA1c", "min": 4.0, "max": 5.7, "unit": "%", "gender": "all", "description": "3-month average blood sugar. 5.7–6.4% is pre-diabetic; 6.5%+ indicates diabetes."},
    {"test": "Post Prandial Glucose", "min": 70.0, "max": 140.0, "unit": "mg/dL", "gender": "all", "description": "Blood sugar 2 hours after eating."},
    # Lipid Profile
    {"test": "Total Cholesterol", "min": 0.0, "max": 200.0, "unit": "mg/dL", "gender": "all", "description": "Overall cholesterol. Above 240 is high risk for heart disease."},
    {"test": "LDL Cholesterol", "min": 0.0, "max": 100.0, "unit": "mg/dL", "gender": "all", "description": "Bad cholesterol. High LDL increases heart disease risk."},
    {"test": "HDL Cholesterol", "min": 40.0, "max": 60.0, "unit": "mg/dL", "gender": "male", "description": "Good cholesterol. Higher HDL is protective against heart disease."},
    {"test": "Triglycerides", "min": 0.0, "max": 150.0, "unit": "mg/dL", "gender": "all", "description": "Blood fats. High levels increase heart disease and pancreatitis risk."},
    # Thyroid
    {"test": "TSH", "min": 0.4, "max": 4.0, "unit": "µIU/mL", "gender": "all", "description": "Thyroid-stimulating hormone. Low TSH = overactive thyroid; high TSH = underactive thyroid."},
    {"test": "T3", "min": 80.0, "max": 200.0, "unit": "ng/dL", "gender": "all", "description": "Triiodothyronine — active thyroid hormone affecting metabolism."},
    {"test": "T4", "min": 5.0, "max": 12.0, "unit": "µg/dL", "gender": "all", "description": "Thyroxine — main thyroid hormone controlling metabolism."},
]


def build_vector_store(persist_path: str | None = None) -> chromadb.Collection:
    """
    Build or load the ChromaDB vector store with medical benchmarks.

    Args:
        persist_path: Directory to persist ChromaDB. Uses config default if None.

    Returns:
        ChromaDB collection ready for querying.
    """
    db_path = persist_path or config.CHROMA_DB_PATH
    Path(db_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"Initialising ChromaDB at: {db_path}")

    client = chromadb.PersistentClient(path=db_path)

    # Use HuggingFace sentence-transformers for embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Fast, lightweight, good accuracy
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="medical_benchmarks",
        embedding_function=ef,
        metadata={"description": "Lab test normal ranges and clinical descriptions"},
    )

    # Only populate if empty
    if collection.count() == 0:
        logger.info("Populating vector store with medical benchmarks...")
        _populate_collection(collection)
        logger.info(f"✅ Loaded {collection.count()} benchmark documents")
    else:
        logger.info(f"Vector store already populated with {collection.count()} documents")

    return collection


def _populate_collection(collection: chromadb.Collection):
    """Insert all benchmark records into ChromaDB."""
    documents, metadatas, ids = [], [], []

    for i, bench in enumerate(MEDICAL_BENCHMARKS):
        # Document text used for embedding / semantic search
        doc_text = (
            f"{bench['test']} normal range is {bench['min']} to {bench['max']} {bench['unit']}. "
            f"{bench['description']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "test": bench["test"].lower(),
            "min": bench["min"],
            "max": bench["max"],
            "unit": bench["unit"],
            "gender": bench.get("gender", "all"),
            "description": bench["description"],
        })
        ids.append(f"bench_{i:03d}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_benchmark(collection: chromadb.Collection, test_name: str, n_results: int = 3) -> list[dict]:
    """
    Retrieve normal range benchmarks for a given lab test name.

    Args:
        collection: ChromaDB collection.
        test_name: Name of the lab test (e.g. "Hemoglobin").
        n_results: Max number of matching benchmarks to return.

    Returns:
        List of benchmark dicts with min, max, unit, description.
    """
    results = collection.query(
        query_texts=[test_name],
        n_results=min(n_results, collection.count()),
    )

    benchmarks = []
    if results and results["metadatas"]:
        for meta in results["metadatas"][0]:
            benchmarks.append(meta)

    return benchmarks
