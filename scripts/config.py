"""
MedAgent 360 · Shared Configuration
Loads environment variables and exposes typed config across all modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ──────────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ── Twilio ────────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "")
    DOCTOR_PHONE: str = os.getenv("DOCTOR_PHONE", "")

    # ── HuggingFace ───────────────────────────────────────────────────────────
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/medagent.db")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./lab_report/data/chroma_db")

    # ── ngrok ─────────────────────────────────────────────────────────────────
    NGROK_TUNNEL_URL: str = os.getenv("NGROK_TUNNEL_URL", "")

    @classmethod
    def validate(cls) -> list[str]:
        """Returns list of missing critical env vars."""
        missing = []
        if not cls.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        return missing


config = Config()
