"""
MedAgent 360 · FastAPI Backend
REST API layer connecting all three modules.
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("medagent360.api")

app = FastAPI(
    title="MedAgent 360 API",
    description="Autonomous Healthcare AI Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "project": "MedAgent 360",
        "version": "1.0.0",
        "modules": ["lab_report", "prescription", "followup"],
        "status": "running",
    }


@app.get("/health")
def health():
    missing = config.validate()
    return {
        "status": "ok" if not missing else "degraded",
        "missing_env_vars": missing,
    }


# ── Module A: Lab Report ──────────────────────────────────────────────────────

@app.post("/analyze-lab")
async def analyze_lab_report(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
):
    """Upload a lab report PDF and get AI analysis."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from lab_report.rag_pipeline import run_full_pipeline
        result = run_full_pipeline(tmp_path, language=language)
        return result
    except Exception as e:
        logger.error(f"Lab report analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Module B: Prescription (placeholder) ─────────────────────────────────────

@app.post("/parse-prescription")
async def parse_prescription(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
):
    """Upload a prescription image and get structured medicine info."""
    # Module B implementation — Dev 2
    return {"message": "Module B coming soon", "language": language}


# ── Module C: Follow-up Agent (placeholder) ───────────────────────────────────

@app.post("/checkin")
async def checkin_webhook(data: dict):
    """Twilio WhatsApp webhook endpoint for patient check-ins."""
    # Module C implementation — Dev 3
    return {"message": "Module C coming soon"}
