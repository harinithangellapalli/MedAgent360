"""
MedAgent 360 · Module C · Steps C1–C7
Autonomous Follow-up Agent
C1: Twilio WhatsApp webhook (FastAPI)
C2: APScheduler daily check-in sender
C3: Webhook response handler + SQLite storage
C4: LLM symptom analysis (Normal/Concerning/Critical)
C5: Doctor alert engine
C6: Recovery tracker
C7: Full loop orchestration
"""

import json
import sqlite3
import re
from datetime import datetime, date
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from twilio.rest import Client as TwilioClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("followup.agent")

# Severity levels returned by LLM
SEVERITY_LEVELS = {
    "NORMAL":      {"icon": "✅", "action": "log_only",      "color": "#00cc44"},
    "CONCERNING":  {"icon": "⚠️", "action": "flag_review",   "color": "#ff9900"},
    "CRITICAL":    {"icon": "🚨", "action": "alert_doctor",  "color": "#ff4444"},
}

# Check-in message templates
CHECKIN_TEMPLATES = {
    "English": (
        "Hello {name}! 👋 This is MedAgent 360, your health assistant.\n\n"
        "How are you feeling today? Please reply with:\n"
        "• Your current symptoms (if any)\n"
        "• Pain level (0-10)\n"
        "• Any medicine side effects\n\n"
        "Your reply helps us monitor your recovery. 🏥"
    ),
    "Telugu": (
        "నమస్కారం {name}! 👋 నేను MedAgent 360, మీ ఆరోగ్య సహాయకుడు.\n\n"
        "ఈరోజు మీరు ఎలా అనిపిస్తున్నారు? దయచేసి మీ:\n"
        "• ప్రస్తుత లక్షణాలు (ఏమైనా ఉంటే)\n"
        "• నొప్పి స్థాయి (0-10)\n"
        "• ఏదైనా మందుల దుష్ప్రభావాలు\n\n"
        "మీ సమాధానం మీ కోలుకోవడాన్ని పర్యవేక్షించడానికి సహాయపడుతుంది. 🏥"
    ),
    "Hindi": (
        "नमस्ते {name}! 👋 मैं MedAgent 360, आपका स्वास्थ्य सहायक हूँ।\n\n"
        "आज आप कैसा महसूस कर रहे हैं? कृपया बताएं:\n"
        "• आपके वर्तमान लक्षण (यदि कोई हो)\n"
        "• दर्द का स्तर (0-10)\n"
        "• कोई दवा के दुष्प्रभाव\n\n"
        "आपका जवाब हमें आपकी रिकवरी पर नज़र रखने में मदद करता है। 🏥"
    ),
}


# ── Database Setup ────────────────────────────────────────────────────────────

def init_database(db_path: str | None = None):
    """Initialize all Module C SQLite tables."""
    db_path = db_path or config.SQLITE_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            name TEXT DEFAULT 'Patient',
            language TEXT DEFAULT 'English',
            doctor_phone TEXT,
            enrolled_at TEXT,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS checkin_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone TEXT NOT NULL,
            message_sid TEXT,
            sent_at TEXT,
            responded INTEGER DEFAULT 0,
            response_text TEXT,
            response_at TEXT
        );

        CREATE TABLE IF NOT EXISTS symptom_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone TEXT NOT NULL,
            response_text TEXT,
            severity TEXT,
            symptoms_identified TEXT,
            pain_level INTEGER,
            alert_sent INTEGER DEFAULT 0,
            analyzed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS doctor_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone TEXT NOT NULL,
            doctor_phone TEXT NOT NULL,
            alert_message TEXT,
            severity TEXT,
            sent_at TEXT,
            message_sid TEXT
        );

        CREATE TABLE IF NOT EXISTS recovery_tracker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone TEXT NOT NULL,
            track_date TEXT NOT NULL,
            severity TEXT,
            pain_level INTEGER,
            symptoms TEXT,
            notes TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("✅ Module C database initialized")


# ── C1: Twilio Setup ──────────────────────────────────────────────────────────

def get_twilio_client() -> TwilioClient:
    if not config.TWILIO_ACCOUNT_SID or not config.TWILIO_AUTH_TOKEN:
        raise ValueError("Twilio credentials not set in .env")
    return TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)


# ── C2: Scheduled Check-in Sender ────────────────────────────────────────────

def send_checkin_message(patient_phone: str, patient_name: str = "Patient", language: str = "English") -> dict:
    """
    Send a WhatsApp check-in message to a patient via Twilio.
    """
    template = CHECKIN_TEMPLATES.get(language, CHECKIN_TEMPLATES["English"])
    message_body = template.format(name=patient_name)

    try:
        client = get_twilio_client()
        msg = client.messages.create(
            from_=config.TWILIO_WHATSAPP_FROM,
            to=patient_phone if patient_phone.startswith("whatsapp:") else f"whatsapp:{patient_phone}",
            body=message_body,
        )

        # Log to DB
        _log_checkin_sent(patient_phone, msg.sid)
        logger.info(f"  ✅ Check-in sent to {patient_phone} | SID: {msg.sid}")
        return {"success": True, "message_sid": msg.sid, "patient_phone": patient_phone}

    except Exception as e:
        logger.error(f"  ❌ Failed to send check-in to {patient_phone}: {e}")
        return {"success": False, "error": str(e), "patient_phone": patient_phone}


def send_daily_checkins():
    """APScheduler job — runs at 8:00 AM daily, sends to all active patients."""
    logger.info("⏰ Running daily check-in job...")
    db_path = config.SQLITE_DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        patients = conn.execute(
            "SELECT phone, name, language FROM patients WHERE active = 1"
        ).fetchall()
        conn.close()
    except Exception as e:
        logger.error(f"DB read failed: {e}")
        return

    results = []
    for phone, name, language in patients:
        result = send_checkin_message(phone, name, language)
        results.append(result)

    sent = sum(1 for r in results if r.get("success"))
    logger.info(f"  Daily check-ins: {sent}/{len(results)} sent successfully")


def start_scheduler() -> BackgroundScheduler:
    """Start APScheduler with the daily check-in job."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        send_daily_checkins,
        trigger=CronTrigger(hour=8, minute=0),
        id="daily_checkin",
        name="Daily Patient Check-in",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("🕗 APScheduler started — daily check-ins at 08:00 AM")
    return scheduler


# ── C3: Webhook Response Handler ─────────────────────────────────────────────

def handle_patient_response(patient_phone: str, response_text: str) -> dict:
    """
    Called when Twilio webhook fires (patient replied on WhatsApp).
    Stores response and triggers analysis.
    """
    logger.info(f"📨 Response from {patient_phone}: {response_text[:80]}...")

    # Store raw response
    _log_checkin_response(patient_phone, response_text)

    # C4: Analyze
    analysis = analyze_symptoms(patient_phone, response_text)

    # C5: Alert if critical
    alert_result = None
    if analysis["severity"] == "CRITICAL":
        alert_result = send_doctor_alert(patient_phone, analysis)

    # C6: Update recovery tracker
    update_recovery_tracker(patient_phone, analysis)

    return {
        "patient_phone": patient_phone,
        "analysis": analysis,
        "alert_sent": alert_result is not None,
        "alert_result": alert_result,
    }


# ── C4: LLM Symptom Analysis ─────────────────────────────────────────────────

def analyze_symptoms(patient_phone: str, response_text: str) -> dict:
    """
    Use Gemini to classify patient's WhatsApp reply.
    Returns: severity (NORMAL/CONCERNING/CRITICAL), symptoms, pain_level.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.1,
    )

    prompt = f"""
A patient sent this WhatsApp reply to their health check-in:
---
{response_text}
---

Analyze and return ONLY valid JSON:
{{
  "severity": "NORMAL" | "CONCERNING" | "CRITICAL",
  "symptoms": ["list", "of", "identified", "symptoms"],
  "pain_level": 0-10 or null,
  "reasoning": "Brief explanation (1 sentence)",
  "urgent_keywords": ["any", "alarming", "phrases"]
}}

Severity rules:
- CRITICAL: chest pain, severe breathing difficulty, unconscious, stroke signs, pain > 8, severe bleeding, suicidal thoughts
- CONCERNING: moderate pain (5-7), persistent fever, vomiting, confusion, worsening symptoms
- NORMAL: mild symptoms, feeling good, minor discomfort, pain < 4

Be conservative — when in doubt, classify higher.
"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a triage assistant. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = re.sub(r"```json\s*|\s*```", "", response.content.strip())
        result = json.loads(raw)
        severity = result.get("severity", "CONCERNING")

    except Exception as e:
        logger.error(f"Symptom analysis failed: {e}")
        result = {"severity": "CONCERNING", "symptoms": [], "pain_level": None, "reasoning": "Analysis failed"}
        severity = "CONCERNING"

    # Store to DB
    _log_analysis(patient_phone, response_text, result)
    severity_meta = SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS["CONCERNING"])
    logger.info(f"  Analysis: {severity_meta['icon']} {severity} | {result.get('reasoning', '')}")

    return {**result, "severity_icon": severity_meta["icon"], "severity_color": severity_meta["color"]}


# ── C5: Doctor Alert Engine ───────────────────────────────────────────────────

def send_doctor_alert(patient_phone: str, analysis: dict) -> dict | None:
    """
    Send urgent SMS/WhatsApp alert to the doctor when patient is CRITICAL.
    """
    doctor_phone = config.DOCTOR_PHONE
    if not doctor_phone:
        logger.warning("DOCTOR_PHONE not set — skipping alert")
        return None

    patient_name = _get_patient_name(patient_phone)
    symptoms_str = ", ".join(analysis.get("symptoms", [])) or "Not specified"
    pain = analysis.get("pain_level", "N/A")
    reasoning = analysis.get("reasoning", "")

    alert_msg = (
        f"🚨 MEDAGENT 360 CRITICAL ALERT 🚨\n\n"
        f"Patient: {patient_name}\n"
        f"Phone: {patient_phone}\n"
        f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        f"Symptoms: {symptoms_str}\n"
        f"Pain Level: {pain}/10\n"
        f"Assessment: {reasoning}\n\n"
        f"⚠️ Immediate attention required."
    )

    try:
        client = get_twilio_client()
        # Send to doctor via WhatsApp if available, else SMS
        to_number = doctor_phone if doctor_phone.startswith("whatsapp:") else doctor_phone
        msg = client.messages.create(
            from_=config.TWILIO_WHATSAPP_FROM,
            to=to_number,
            body=alert_msg,
        )

        _log_doctor_alert(patient_phone, doctor_phone, alert_msg, analysis["severity"], msg.sid)
        logger.info(f"  🚨 Doctor alert sent | SID: {msg.sid}")
        return {"success": True, "message_sid": msg.sid, "doctor_phone": doctor_phone}

    except Exception as e:
        logger.error(f"  Doctor alert FAILED: {e}")
        return {"success": False, "error": str(e)}


# ── C6: Recovery Tracker ──────────────────────────────────────────────────────

def update_recovery_tracker(patient_phone: str, analysis: dict):
    """Log daily symptom data to recovery timeline."""
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO recovery_tracker
        (patient_phone, track_date, severity, pain_level, symptoms, notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_phone,
        str(date.today()),
        analysis.get("severity"),
        analysis.get("pain_level"),
        json.dumps(analysis.get("symptoms", [])),
        analysis.get("reasoning", ""),
    ))
    conn.commit()
    conn.close()


def get_recovery_timeline(patient_phone: str, days: int = 14) -> list[dict]:
    """Fetch last N days of recovery data for dashboard chart."""
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        rows = conn.execute("""
            SELECT track_date, severity, pain_level, symptoms
            FROM recovery_tracker
            WHERE patient_phone = ?
            ORDER BY track_date DESC
            LIMIT ?
        """, (patient_phone, days)).fetchall()
        conn.close()
        return [
            {"date": r[0], "severity": r[1], "pain_level": r[2], "symptoms": json.loads(r[3] or "[]")}
            for r in rows
        ]
    except Exception:
        return []


def enroll_patient(phone: str, name: str, language: str = "English", doctor_phone: str = "") -> dict:
    """Register a new patient for follow-up monitoring."""
    init_database()
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO patients (phone, name, language, doctor_phone, enrolled_at)
            VALUES (?, ?, ?, ?, ?)
        """, (phone, name, language, doctor_phone, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        logger.info(f"✅ Enrolled patient: {name} ({phone})")
        return {"success": True, "phone": phone, "name": name}
    except Exception as e:
        logger.error(f"Enroll failed: {e}")
        return {"success": False, "error": str(e)}


# ── DB Helpers ─────────────────────────────────────────────────────────────────

def _log_checkin_sent(patient_phone: str, message_sid: str):
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.execute(
            "INSERT INTO checkin_messages (patient_phone, message_sid, sent_at) VALUES (?, ?, ?)",
            (patient_phone, message_sid, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB log failed: {e}")


def _log_checkin_response(patient_phone: str, response_text: str):
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.execute("""
            UPDATE checkin_messages SET responded=1, response_text=?, response_at=?
            WHERE patient_phone=? AND responded=0
            ORDER BY sent_at DESC LIMIT 1
        """, (response_text, datetime.now().isoformat(), patient_phone))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Response log failed: {e}")


def _log_analysis(patient_phone: str, response_text: str, analysis: dict):
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.execute("""
            INSERT INTO symptom_analysis
            (patient_phone, response_text, severity, symptoms_identified, pain_level, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_phone, response_text,
            analysis.get("severity"),
            json.dumps(analysis.get("symptoms", [])),
            analysis.get("pain_level"),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Analysis log failed: {e}")


def _log_doctor_alert(patient_phone, doctor_phone, msg, severity, sid):
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        conn.execute("""
            INSERT INTO doctor_alerts
            (patient_phone, doctor_phone, alert_message, severity, sent_at, message_sid)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_phone, doctor_phone, msg, severity, datetime.now().isoformat(), sid))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Alert log failed: {e}")


def _get_patient_name(phone: str) -> str:
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH)
        row = conn.execute("SELECT name FROM patients WHERE phone = ?", (phone,)).fetchone()
        conn.close()
        return row[0] if row else phone
    except Exception:
        return phone
