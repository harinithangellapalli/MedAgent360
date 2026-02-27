"""
MedAgent 360 · Module A · Steps A3–A5 (Phase 1 Enhanced)
RAG Pipeline + Gemini LLM Analysis
- Richer classification with reasoning per parameter
- Stronger multilingual summaries (Telugu / Hindi / English)
- Structured JSON output for dashboard rendering
- Critical alert detection for Module C integration
"""

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from lab_report.vector_store import build_vector_store, query_benchmark
from scripts.logger import get_logger
from scripts.config import config

logger = get_logger("lab_report.rag_pipeline")

RISK_LEVELS = {
    "NORMAL":   {"icon": "✅", "color": "#00cc44", "priority": 0},
    "LOW":      {"icon": "⬇️", "color": "#ffcc00", "priority": 1},
    "HIGH":     {"icon": "⬆️", "color": "#ff9900", "priority": 2},
    "CRITICAL": {"icon": "🚨", "color": "#ff4444", "priority": 3},
    "UNKNOWN":  {"icon": "❓", "color": "#aaaaaa", "priority": 0},
}

LANGUAGE_PROMPTS = {
    "English": {
        "system": "You are a compassionate medical assistant explaining lab results to a rural Indian patient in simple English.",
        "instruction": "Write in clear, simple English. Avoid jargon.",
    },
    "Telugu": {
        "system": "మీరు గ్రామీణ భారత రోగికి ల్యాబ్ ఫలితాలను సరళమైన తెలుగులో వివరించే వైద్య సహాయకుడు.",
        "instruction": "తెలుగులో రాయండి. సరళమైన పదాలు వాడండి. వైద్య పరిభాష తప్పించుకోండి.",
    },
    "Hindi": {
        "system": "आप एक सहानुभूतिपूर्ण चिकित्सा सहायक हैं जो ग्रामीण भारतीय मरीज को सरल हिंदी में लैब परिणाम समझा रहे हैं।",
        "instruction": "सरल हिंदी में लिखें। मेडिकल शब्दजाल से बचें।",
    },
}


def _get_llm() -> ChatGoogleGenerativeAI:
    if not config.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in .env")
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
    )


# ── Step A3 + A4: Classification ─────────────────────────────────────────────

def classify_lab_values(lab_values: list[dict]) -> list[dict]:
    """
    For each lab value: query ChromaDB for normal range → classify → add Gemini reasoning.

    Returns enriched list with status, risk metadata, and plain-language reason.
    """
    logger.info("🔍 Loading vector store for benchmark lookup...")
    collection = build_vector_store()

    enriched = []
    for item in lab_values:
        test = item.get("test", "").strip()
        raw_value = item.get("value", "N/A")
        numeric_value = _parse_numeric(raw_value)

        # Query benchmark
        benchmarks = query_benchmark(collection, test, n_results=1)

        if benchmarks and numeric_value is not None:
            bench = benchmarks[0]
            b_min = float(bench["min"])
            b_max = float(bench["max"])
            status = _classify(numeric_value, b_min, b_max)
            risk_meta = RISK_LEVELS[status]

            enriched.append({
                **item,
                "numeric_value": numeric_value,
                "status": status,
                "risk_icon": risk_meta["icon"],
                "risk_color": risk_meta["color"],
                "risk_priority": risk_meta["priority"],
                "benchmark_min": b_min,
                "benchmark_max": b_max,
                "benchmark_unit": bench.get("unit", ""),
                "benchmark_description": bench.get("description", ""),
                "deviation_pct": _deviation_pct(numeric_value, b_min, b_max),
            })
        else:
            enriched.append({
                **item,
                "numeric_value": numeric_value,
                "status": "UNKNOWN",
                "risk_icon": RISK_LEVELS["UNKNOWN"]["icon"],
                "risk_color": RISK_LEVELS["UNKNOWN"]["color"],
                "risk_priority": 0,
                "benchmark_min": None,
                "benchmark_max": None,
                "benchmark_unit": item.get("unit", ""),
                "benchmark_description": "",
                "deviation_pct": None,
            })

    # Sort: CRITICAL → HIGH → LOW → NORMAL → UNKNOWN
    enriched.sort(key=lambda x: x["risk_priority"], reverse=True)

    critical = sum(1 for x in enriched if x["status"] == "CRITICAL")
    abnormal = sum(1 for x in enriched if x["status"] in ("HIGH", "LOW"))
    logger.info(f"  🚨 CRITICAL: {critical} | ⚠️ ABNORMAL: {abnormal} | Total: {len(enriched)}")

    return enriched


# ── Step A5: Multilingual Summary ─────────────────────────────────────────────

def generate_summary(
    classified_values: list[dict],
    patient_info: dict,
    language: str = "English",
    raw_text: str = "",
) -> dict:
    """
    Generate a rich, patient-friendly summary using Gemini.
    Produces structured output with headline, findings, and recommendations.
    """
    llm = _get_llm()
    lang_config = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["English"])

    # Build findings context
    abnormal = [v for v in classified_values if v["status"] in ("CRITICAL", "HIGH", "LOW")]
    normal = [v for v in classified_values if v["status"] == "NORMAL"]

    if classified_values:
        findings_text = "\n".join([
            f"- {v['test']}: {v['value']} {v.get('unit','')} "
            f"[{v['status']}] (normal: {v['benchmark_min']}–{v['benchmark_max']} {v['benchmark_unit']})"
            for v in classified_values if v["status"] != "UNKNOWN"
        ])
    else:
        findings_text = f"No structured tables found. Please analyze the following raw text from the report:\n{raw_text}"

    patient_name = patient_info.get("name", "the patient")
    patient_age = patient_info.get("age", "unknown")
    patient_gender = patient_info.get("gender", "unknown")

    prompt = f"""
Patient: {patient_name}, Age: {patient_age}, Gender: {patient_gender}

Lab Findings:
{findings_text}

Your task — write a patient summary in exactly this structure:

1. HEADLINE (1 sentence): Overall health snapshot
2. KEY FINDINGS (2-4 bullet points): Mention abnormal values and what they mean in simple terms
3. GOOD NEWS (1 sentence): Mention what's normal
4. RECOMMENDATIONS (2-3 actionable steps): What should the patient do next?

{lang_config['instruction']}
Be warm, clear, and reassuring. Do not use medical abbreviations without explanation.
Length: 150-200 words total.
"""

    system_msg = lang_config["system"]
    logger.info(f"🤖 Generating Gemini summary in {language}...")

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt),
    ])
    summary_text = response.content.strip()

    # Gemini reasoning for critical values
    critical_explanations = {}
    critical_items = [v for v in classified_values if v["status"] == "CRITICAL"]
    if critical_items:
        critical_explanations = _generate_critical_explanations(llm, critical_items, language)

    return {
        "summary": summary_text,
        "critical_flags": [v["test"] for v in critical_items],
        "critical_explanations": critical_explanations,
        "language": language,
        "total_tests": len(classified_values),
        "normal_count": len(normal),
        "abnormal_count": len(abnormal),
        "critical_count": len(critical_items),
    }


def _generate_critical_explanations(llm, critical_items: list[dict], language: str) -> dict:
    """Generate a short plain-language explanation for each critical value."""
    explanations = {}
    lang_config = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["English"])

    for item in critical_items:
        prompt = (
            f"The patient's {item['test']} is {item['value']} {item.get('unit','')}. "
            f"Normal range is {item['benchmark_min']}–{item['benchmark_max']} {item['benchmark_unit']}. "
            f"In 1-2 simple sentences, explain what this means for the patient and why it's urgent. "
            f"{lang_config['instruction']}"
        )
        try:
            resp = llm.invoke([
                SystemMessage(content=lang_config["system"]),
                HumanMessage(content=prompt),
            ])
            explanations[item["test"]] = resp.content.strip()
        except Exception as e:
            logger.warning(f"Could not generate explanation for {item['test']}: {e}")
            explanations[item["test"]] = item.get("benchmark_description", "")

    return explanations


# ── Master Pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(pdf_path: str, language: str = "English") -> dict:
    """
    Master pipeline: PDF → parse → classify → summarise → voice-ready output.
    Returns complete dict ready for Streamlit + FastAPI.
    """
    from lab_report.pdf_parser import extract_lab_values

    logger.info(f"🔬 Starting full pipeline | file={pdf_path} | lang={language}")

    # A1: Parse
    parsed = extract_lab_values(pdf_path)

    # A3–A4: Classify
    classified = classify_lab_values(parsed["lab_values"])

    # A5: Summarise
    summary = generate_summary(classified, parsed["patient_info"], language, raw_text=parsed.get("raw_text", ""))

    return {
        "patient_info": parsed["patient_info"],
        "classified_values": classified,
        "summary": summary["summary"],
        "critical_flags": summary["critical_flags"],
        "critical_explanations": summary.get("critical_explanations", {}),
        "language": language,
        "extraction_method": parsed.get("extraction_method", "unknown"),
        "stats": {
            "total":    summary["total_tests"],
            "normal":   summary["normal_count"],
            "abnormal": summary["abnormal_count"],
            "critical": summary["critical_count"],
            "pages":    parsed.get("pages", 1),
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_numeric(value: str) -> float | None:
    if not value:
        return None
    match = re.search(r"[\d]+\.?[\d]*", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _classify(value: float, b_min: float, b_max: float) -> str:
    if value < b_min * 0.65 or value > b_max * 1.6:
        return "CRITICAL"
    elif value < b_min:
        return "LOW"
    elif value > b_max:
        return "HIGH"
    return "NORMAL"


def _deviation_pct(value: float, b_min: float, b_max: float) -> float | None:
    """How far outside normal range (%) — 0 if normal."""
    if value < b_min:
        return round((b_min - value) / b_min * 100, 1)
    elif value > b_max:
        return round((value - b_max) / b_max * 100, 1)
    return 0.0
