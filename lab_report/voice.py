"""
MedAgent 360 · Module A · Step A6
Voice Output Generator
Converts the AI summary text into MP3 audio using gTTS.
Supports English, Telugu, and Hindi.
"""

from gtts import gTTS
from pathlib import Path
import tempfile
from scripts.logger import get_logger

logger = get_logger("lab_report.voice")

# gTTS language codes
LANGUAGE_CODES = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
}


def generate_audio(text: str, language: str = "English", output_path: str | None = None) -> str:
    """
    Convert summary text to an MP3 audio file.

    Args:
        text: The summary text to convert.
        language: "English", "Telugu", or "Hindi".
        output_path: Where to save the MP3. If None, saves to a temp file.

    Returns:
        Path to the generated MP3 file.
    """
    lang_code = LANGUAGE_CODES.get(language, "en")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = tmp.name

    logger.info(f"Generating {language} audio → {output_path}")

    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(output_path)

    logger.info(f"✅ Audio saved: {output_path}")
    return output_path


def generate_section_audios(classified_values: list[dict], language: str = "English") -> list[dict]:
    """
    Generate short audio clips for critical/abnormal findings only.
    Useful for per-parameter audio in the Streamlit dashboard.

    Returns:
        List of {"test", "status", "audio_path"} dicts.
    """
    audio_items = []
    for item in classified_values:
        if item["status"] in ("CRITICAL", "HIGH", "LOW"):
            direction = "high" if item["status"] in ("HIGH", "CRITICAL") else "low"
            status_label = item["status"]

            if language == "Hindi":
                msg = f"{item['test']} का स्तर {direction} है। कृपया डॉक्टर से मिलें।"
            elif language == "Telugu":
                msg = f"{item['test']} స్థాయి {direction}గా ఉంది. దయచేసి వైద్యుడిని సంప్రదించండి."
            else:
                msg = f"{item['test']} is {status_label}. Your value is {item['value']} {item['unit']}, normal range is {item['benchmark_min']} to {item['benchmark_max']}."

            audio_path = generate_audio(msg, language)
            audio_items.append({
                "test": item["test"],
                "status": item["status"],
                "audio_path": audio_path,
                "message": msg,
            })

    return audio_items
