"""
MedAgent 360 · Module B · Steps B3–B6
Prescription Parser Pipeline
B3: LLM extracts medicines, dosage, frequency, duration
B4: IndicTrans translation to Telugu/Hindi
B5: gTTS audio per medicine
B6: APScheduler medication reminders
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from gtts import gTTS
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("prescription.parser")

# ── Translation Maps (fallback if IndicTrans unavailable) ─────────────────────
COMMON_MEDICINE_TRANSLATIONS = {
    "Telugu": {
        "Take": "తీసుకోండి", "tablet": "మాత్ర", "capsule": "క్యాప్సూల్",
        "morning": "ఉదయం", "evening": "సాయంత్రం", "night": "రాత్రి",
        "after food": "భోజనం తర్వాత", "before food": "భోజనానికి ముందు",
        "with water": "నీళ్ళతో", "days": "రోజులు", "weeks": "వారాలు",
        "times": "సార్లు", "daily": "రోజూ",
    },
    "Hindi": {
        "Take": "लें", "tablet": "गोली", "capsule": "कैप्सूल",
        "morning": "सुबह", "evening": "शाम", "night": "रात",
        "after food": "खाने के बाद", "before food": "खाने से पहले",
        "with water": "पानी के साथ", "days": "दिन", "weeks": "हफ्ते",
        "times": "बार", "daily": "रोज़",
    },
}


def _get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.1,
    )


# ── B3: LLM Medicine Extraction ───────────────────────────────────────────────

def extract_medicines_from_ocr(ocr_text: str) -> list[dict]:
    """
    Use Gemini to parse OCR text and extract structured medicine info.

    Returns list of:
    {
        "medicine": str,
        "dosage": str,       # e.g. "500mg"
        "frequency": str,    # e.g. "twice daily"
        "timing": str,       # e.g. "after food"
        "duration": str,     # e.g. "5 days"
        "special_notes": str # e.g. "avoid alcohol"
    }
    """
    llm = _get_llm()

    prompt = f"""
You are a pharmacist assistant. Below is OCR-extracted text from a medical prescription.
Extract ALL medicines mentioned and return a JSON array.

OCR Text:
---
{ocr_text}
---

Return ONLY a valid JSON array. No explanation, no markdown. Example:
[
  {{
    "medicine": "Paracetamol 500mg",
    "dosage": "500mg",
    "frequency": "3 times daily",
    "timing": "after food",
    "duration": "5 days",
    "special_notes": ""
  }}
]

Rules:
- If a field is unclear, use "" (empty string)
- Normalize frequency: use "once daily", "twice daily", "3 times daily"
- Include ALL medicines even if info is partial
- Correct obvious OCR errors in medicine names
"""

    logger.info("💊 Extracting medicines with Gemini...")
    response = llm.invoke([
        SystemMessage(content="You are a precise pharmacist assistant. Return only valid JSON."),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    # Strip markdown code blocks if present
    raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

    try:
        medicines = json.loads(raw)
        if not isinstance(medicines, list):
            medicines = [medicines]
        logger.info(f"  Extracted {len(medicines)} medicines")
        return medicines
    except json.JSONDecodeError as e:
        logger.error(f"  JSON parse failed: {e}\nRaw: {raw[:300]}")
        return _fallback_medicine_extraction(ocr_text)


def _fallback_medicine_extraction(text: str) -> list[dict]:
    """Regex fallback if LLM JSON parsing fails."""
    medicines = []
    # Common medicine patterns
    pattern = re.compile(
        r"([A-Z][a-zA-Z\s]+(?:\d+mg|\d+ml)?)\s*[-–]?\s*"
        r"(\d+\s*(?:tablet|tab|cap|capsule|ml)s?)?",
        re.IGNORECASE
    )
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        if len(name) > 3 and not name.lower().startswith(("date", "name", "age", "doctor")):
            medicines.append({
                "medicine": name,
                "dosage": m.group(2) or "",
                "frequency": "",
                "timing": "",
                "duration": "",
                "special_notes": "",
            })
    return medicines[:10]  # cap at 10


# ── B4: Translation ───────────────────────────────────────────────────────────

def translate_prescription(medicines: list[dict], target_language: str) -> list[dict]:
    """
    Translate medicine instructions to Telugu or Hindi.
    Uses Gemini for natural translation (IndicTrans as upgrade path).
    """
    if target_language == "English":
        for m in medicines:
            m["instruction_translated"] = _build_instruction(m, "English")
            m["language"] = "English"
        return medicines

    llm = _get_llm()
    translated = []

    for med in medicines:
        instruction_en = _build_instruction(med, "English")

        lang_prompts = {
            "Telugu": f"Translate this medicine instruction to Telugu script: '{instruction_en}'",
            "Hindi":  f"Translate this medicine instruction to Hindi script: '{instruction_en}'",
        }
        prompt = lang_prompts.get(target_language, lang_prompts["Hindi"])
        prompt += "\nReturn ONLY the translated text. No explanation."

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            translated_text = response.content.strip()
        except Exception as e:
            logger.warning(f"Translation failed for {med['medicine']}: {e}")
            translated_text = instruction_en  # fallback to English

        translated.append({
            **med,
            "instruction_english": instruction_en,
            "instruction_translated": translated_text,
            "language": target_language,
        })
        logger.info(f"  Translated: {med['medicine']} → {target_language}")

    return translated


def _build_instruction(med: dict, language: str) -> str:
    """Build a human-readable instruction string from medicine dict."""
    parts = [f"Take {med.get('medicine', 'this medicine')}"]
    if med.get("dosage"):
        parts[0] += f" ({med['dosage']})"
    if med.get("frequency"):
        parts.append(med["frequency"])
    if med.get("timing"):
        parts.append(med["timing"])
    if med.get("duration"):
        parts.append(f"for {med['duration']}")
    if med.get("special_notes"):
        parts.append(med["special_notes"])
    return " — ".join(parts) + "."


# ── B5: Voice Output ──────────────────────────────────────────────────────────

LANG_CODES = {"English": "en", "Telugu": "te", "Hindi": "hi"}

def generate_medicine_audio(medicines: list[dict], language: str = "English") -> list[dict]:
    """
    Generate MP3 audio for each medicine instruction.
    Returns medicines with "audio_path" added.
    """
    import tempfile
    lang_code = LANG_CODES.get(language, "en")

    for med in medicines:
        text = med.get("instruction_translated") or med.get("instruction_english", "")
        if not text:
            continue
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="rx_")
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(tmp.name)
            med["audio_path"] = tmp.name
            logger.info(f"  🔊 Audio: {med['medicine']} → {tmp.name}")
        except Exception as e:
            logger.warning(f"  Audio failed for {med['medicine']}: {e}")
            med["audio_path"] = None

    return medicines


# ── B6: Reminder Scheduler ────────────────────────────────────────────────────

def schedule_reminders(medicines: list[dict], patient_phone: str, db_path: str | None = None) -> list[dict]:
    """
    Store medication reminder schedules in SQLite for the follow-up agent.
    APScheduler picks these up via Module C.
    """
    db_path = db_path or config.SQLITE_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS medication_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone TEXT NOT NULL,
            medicine_name TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT,
            timing TEXT,
            duration_days INTEGER,
            start_date TEXT,
            end_date TEXT,
            reminder_times TEXT,
            created_at TEXT,
            active INTEGER DEFAULT 1
        )
    """)

    scheduled = []
    start_date = datetime.now().date()

    for med in medicines:
        # Parse duration
        duration_days = _parse_duration_days(med.get("duration", ""))
        end_date = start_date + timedelta(days=duration_days) if duration_days else None

        # Build reminder times from frequency
        reminder_times = _frequency_to_times(med.get("frequency", ""))

        cur.execute("""
            INSERT INTO medication_reminders
            (patient_phone, medicine_name, dosage, frequency, timing, duration_days,
             start_date, end_date, reminder_times, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_phone,
            med.get("medicine", "Unknown"),
            med.get("dosage", ""),
            med.get("frequency", ""),
            med.get("timing", ""),
            duration_days,
            str(start_date),
            str(end_date) if end_date else None,
            json.dumps(reminder_times),
            datetime.now().isoformat(),
        ))

        scheduled.append({**med, "reminder_times": reminder_times, "end_date": str(end_date)})
        logger.info(f"  ⏰ Scheduled: {med.get('medicine')} at {reminder_times}")

    conn.commit()
    conn.close()
    return scheduled


def _parse_duration_days(duration_str: str) -> int:
    if not duration_str:
        return 7  # default 7 days
    m = re.search(r"(\d+)\s*(day|week|month)", duration_str.lower())
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        return n if unit == "day" else n * 7 if unit == "week" else n * 30
    return 7


def _frequency_to_times(frequency: str) -> list[str]:
    freq = frequency.lower()
    if "once" in freq or "1" in freq:
        return ["08:00"]
    elif "twice" in freq or "2" in freq:
        return ["08:00", "20:00"]
    elif "three" in freq or "3" in freq:
        return ["08:00", "14:00", "20:00"]
    elif "four" in freq or "4" in freq:
        return ["08:00", "12:00", "16:00", "20:00"]
    return ["08:00"]


# ── Master Pipeline ───────────────────────────────────────────────────────────

def run_prescription_pipeline(
    image_path: str,
    language: str = "English",
    patient_phone: str = "",
    schedule: bool = True,
) -> dict:
    """
    Full B pipeline: image → OCR → LLM parse → translate → voice → schedule.
    """
    from prescription.ocr_engine import auto_detect_and_extract

    logger.info(f"💊 Starting prescription pipeline | file={image_path} | lang={language}")

    # B1+B2: Preprocess + OCR
    ocr_result = auto_detect_and_extract(image_path)

    if not ocr_result["raw_text"].strip():
        raise ValueError("OCR extracted no text from this image. Please use a clearer photo.")

    # B3: LLM extraction
    medicines = extract_medicines_from_ocr(ocr_result["raw_text"])

    if not medicines:
        raise ValueError("No medicines found in prescription. Please check the image quality.")

    # B4: Translate
    medicines = translate_prescription(medicines, language)

    # B5: Audio
    medicines = generate_medicine_audio(medicines, language)

    # B6: Schedule reminders
    if schedule and patient_phone:
        medicines = schedule_reminders(medicines, patient_phone)

    return {
        "medicines": medicines,
        "ocr_text": ocr_result["raw_text"],
        "ocr_confidence": ocr_result["confidence"],
        "detection_mode": ocr_result.get("detection_mode", "printed"),
        "language": language,
        "medicine_count": len(medicines),
    }
