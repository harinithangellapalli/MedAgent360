"""
MedAgent 360 · FastAPI Backend (Phase 1)
All 3 modules wired: /analyze-lab, /parse-prescription, /checkin webhook
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os
from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("medagent360.api")

app = FastAPI(
    title="MedAgent 360 API",
    description="Autonomous Healthcare AI Agent — PS #22 + #23 + #24",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    logger.info("MedAgent 360 API starting up...")
    try:
        from followup.agent import init_database
        init_database()
    except Exception as e:
        logger.warning(f"DB init skipped: {e}")
    try:
        from followup.agent import start_scheduler
        app.state.scheduler = start_scheduler()
    except Exception as e:
        logger.warning(f"Scheduler skipped: {e}")

@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown()

@app.get("/")
def root():
    return {"project": "MedAgent 360", "version": "1.1.0", "status": "running"}

@app.get("/health")
def health():
    missing = config.validate()
    return {"status": "ok" if not missing else "degraded", "missing_env_vars": missing}

@app.post("/analyze-lab")
async def analyze_lab_report(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        from lab_report.rag_pipeline import run_full_pipeline
        return run_full_pipeline(tmp_path, language=language)
    except Exception as e:
        logger.error(f"Lab analysis failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/parse-prescription")
async def parse_prescription(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
    patient_phone: str = Form(default=""),
    schedule_reminders: bool = Form(default=False),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        raise HTTPException(400, "Please upload an image file (JPG, PNG, etc.)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        from prescription.parser import run_prescription_pipeline
        result = run_prescription_pipeline(tmp_path, language=language, patient_phone=patient_phone, schedule=schedule_reminders)
        for med in result.get("medicines", []):
            med.pop("audio_path", None)
        return result
    except Exception as e:
        logger.error(f"Prescription parse failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/checkin/webhook")
async def twilio_webhook(request: Request):
    form = await request.form()
    patient_phone = form.get("From", "")
    body = form.get("Body", "")
    if not patient_phone or not body:
        return JSONResponse({"error": "Missing From or Body"}, status_code=400)
    try:
        from followup.agent import handle_patient_response
        result = handle_patient_response(patient_phone, body)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(500, str(e))

@app.post("/checkin/enroll")
async def enroll_patient(
    phone: str = Form(...),
    name: str = Form(default="Patient"),
    language: str = Form(default="English"),
    doctor_phone: str = Form(default=""),
):
    from followup.agent import enroll_patient
    return enroll_patient(phone, name, language, doctor_phone)

@app.post("/checkin/send")
async def manual_checkin(
    phone: str = Form(...),
    name: str = Form(default="Patient"),
    language: str = Form(default="English"),
):
    from followup.agent import send_checkin_message
    return send_checkin_message(phone, name, language)

@app.get("/checkin/recovery/{phone}")
async def get_recovery(phone: str, days: int = 14):
    from followup.agent import get_recovery_timeline
    return {"phone": phone, "timeline": get_recovery_timeline(phone, days)}

@app.get("/checkin/alerts")
async def get_alerts():
    import sqlite3
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        rows = conn.execute("SELECT * FROM doctor_alerts ORDER BY sent_at DESC LIMIT 20").fetchall()
        conn.close()
        cols = ["id", "patient_phone", "doctor_phone", "alert_message", "severity", "sent_at", "message_sid"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        raise HTTPException(500, str(e))
