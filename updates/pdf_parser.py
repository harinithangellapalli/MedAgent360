"""
MedAgent 360 · Module A · Step A1 (Phase 1 Enhanced)
PDF Lab Report Parser — Multi-format, robust table detection
Handles structured tables, borderless tables, and raw-text fallback.
Works with Apollo, SRL, Thyrocare, Metropolis report formats.
"""

import pdfplumber
import pandas as pd
import re
from pathlib import Path
from scripts.logger import get_logger

logger = get_logger("lab_report.pdf_parser")

# Known Indian lab report header aliases
HEADER_ALIASES = {
    "test":      ["test", "parameter", "investigation", "test name", "description", "analyte"],
    "value":     ["value", "result", "observed", "your value", "observed value", "reading"],
    "unit":      ["unit", "units", "uom"],
    "reference": ["reference", "normal", "range", "ref range", "biological ref", "normal range", "ref interval"],
    "status":    ["status", "flag", "remark", "remarks", "interpretation"],
}

# Lines to skip during text extraction
SKIP_PATTERNS = re.compile(
    r"^(page|report|date|time|printed|lab|doctor|address|phone|email|gender|sex|sample|specimen|barcode|accession|ref\s*by|referred|collected|received|reported|technician|pathologist|authorised|authorized|signature|stamp|www|http)",
    re.IGNORECASE,
)


def extract_lab_values(pdf_path: str) -> dict:
    """
    Extract lab test values from a PDF report.

    Extraction strategy (in order of preference):
    1. Explicit bordered tables (pdfplumber default)
    2. Borderless/whitespace-aligned tables (custom strategy)
    3. Raw-text regex fallback

    Returns:
        {
          "raw_text": str,
          "lab_values": list[dict],  # {test, value, unit, reference, status}
          "patient_info": dict,      # {name, age, gender, date, lab_name}
          "pages": int,
          "extraction_method": str,
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"📄 Parsing PDF: {path.name}")

    raw_text = ""
    lab_values = []
    patient_info = {}
    extraction_method = "unknown"

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  Total pages: {total_pages}")

        for page_num, page in enumerate(pdf.pages, 1):
            logger.info(f"  Processing page {page_num}/{total_pages}")
            page_text = page.extract_text() or ""
            raw_text += page_text + "\n"

            if page_num == 1:
                patient_info = _extract_patient_info(page_text)

            # Strategy 1: Standard bordered tables
            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            })
            for table in tables:
                parsed = _parse_lab_table(table)
                lab_values.extend(parsed)

            # Strategy 2: Borderless tables (text + whitespace alignment)
            if not lab_values:
                tables_text = page.extract_tables({
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "intersection_y_tolerance": 10,
                })
                for table in tables_text:
                    parsed = _parse_lab_table(table)
                    lab_values.extend(parsed)

        # Strategy 3: Raw text regex fallback
        if not lab_values:
            logger.warning("  No tables detected — using text extraction fallback")
            lab_values = _extract_from_text(raw_text)
            extraction_method = "text_regex"
        else:
            extraction_method = "table"

    # Deduplicate by test name
    lab_values = _deduplicate(lab_values)
    logger.info(f"  ✅ Extracted {len(lab_values)} parameters via [{extraction_method}]")

    return {
        "raw_text": raw_text.strip(),
        "lab_values": lab_values,
        "patient_info": patient_info,
        "pages": total_pages,
        "extraction_method": extraction_method,
    }


# ── Patient Info Extraction ───────────────────────────────────────────────────

def _extract_patient_info(text: str) -> dict:
    """Extract patient demographics and lab metadata from header text."""
    info = {
        "name": "Unknown",
        "age": "Unknown",
        "gender": "Unknown",
        "date": "Unknown",
        "lab_name": "Unknown",
    }

    # Name — multiple formats
    for pattern in [
        r"(?:Patient\s*Name|Name)\s*[:\-]?\s*([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){1,3})",
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s*([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){1,3})",
    ]:
        m = re.search(pattern, text)
        if m:
            info["name"] = m.group(1).strip()
            break

    # Age
    m = re.search(r"(?:Age)\s*[:\-]?\s*(\d{1,3})\s*(?:Yrs?\.?|Years?)?", text, re.IGNORECASE)
    if m:
        info["age"] = m.group(1)

    # Gender
    m = re.search(r"(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F)\b", text, re.IGNORECASE)
    if m:
        g = m.group(1).upper()
        info["gender"] = "Male" if g in ("M", "MALE") else "Female"

    # Date (multiple formats)
    m = re.search(r"(?:Date|Collected|Reported)\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})", text, re.IGNORECASE)
    if m:
        info["date"] = m.group(1)

    # Lab name (first all-caps line or known lab names)
    known_labs = ["Apollo", "SRL", "Thyrocare", "Metropolis", "LIMS", "Dr Lal", "Vijaya", "CARE"]
    for lab in known_labs:
        if lab.lower() in text.lower():
            info["lab_name"] = lab
            break

    return info


# ── Table Parsing ─────────────────────────────────────────────────────────────

def _parse_lab_table(table: list) -> list[dict]:
    """
    Parse a pdfplumber table matrix into structured lab value dicts.
    Handles merged cells, multi-line test names, and missing columns.
    """
    results = []
    if not table or len(table) < 2:
        return results

    # Clean table: normalize None → ""
    table = [[str(cell).strip() if cell else "" for cell in row] for row in table]

    # Find header row (first row with recognizable column names)
    header_row_idx = 0
    col_map = {}
    for row_idx, row in enumerate(table[:4]):  # scan first 4 rows for header
        row_lower = [c.lower() for c in row]
        col_map = _map_columns(row_lower)
        if col_map.get("test") is not None and col_map.get("value") is not None:
            header_row_idx = row_idx
            break

    # Positional fallback if no header found
    if col_map.get("test") is None:
        col_map = {"test": 0, "value": 1, "unit": 2, "reference": 3, "status": 4}

    test_col = col_map.get("test", 0)
    value_col = col_map.get("value", 1)
    unit_col = col_map.get("unit", 2)
    ref_col = col_map.get("reference", 3)
    status_col = col_map.get("status", 4)

    for row in table[header_row_idx + 1:]:
        if len(row) <= test_col:
            continue

        test_name = row[test_col].strip()

        # Skip empty rows, section headers, and footer lines
        if not test_name:
            continue
        if SKIP_PATTERNS.match(test_name):
            continue
        if len(test_name) < 2 or test_name.lower() in ("test", "parameter", "investigation", "total"):
            continue
        if re.match(r"^[\-\*\=\_\s]+$", test_name):  # decorative separator lines
            continue

        def safe_get(col):
            if col is not None and col < len(row):
                return row[col].strip()
            return ""

        results.append({
            "test": _clean_test_name(test_name),
            "value": safe_get(value_col),
            "unit": safe_get(unit_col),
            "reference": safe_get(ref_col),
            "status": safe_get(status_col),
        })

    return results


def _map_columns(header_row: list) -> dict:
    """Map column indices from a header row using alias matching."""
    col_map = {}
    for field, aliases in HEADER_ALIASES.items():
        for idx, cell in enumerate(header_row):
            if any(alias in cell for alias in aliases):
                col_map[field] = idx
                break
    return col_map


def _clean_test_name(name: str) -> str:
    """Normalize test names: remove leading numbers, extra whitespace."""
    name = re.sub(r"^\d+[\.\)]\s*", "", name)  # remove "1. " or "1) "
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()


# ── Text Fallback Extraction ──────────────────────────────────────────────────

def _extract_from_text(text: str) -> list[dict]:
    """
    Regex-based extraction for unstructured / text-only PDFs.
    Handles formats like:
      Hemoglobin        13.5    g/dL    13.0-17.0    Normal
      SGPT/ALT  :  42  U/L  (0-40)
    """
    results = []
    lines = text.split("\n")

    # Pattern A: tab/space separated columns
    pattern_cols = re.compile(
        r"^([A-Za-z][A-Za-z\s\(\)\/\-\.]{2,45?}?)\s{2,}"   # test name (2+ spaces separator)
        r"([\d]+\.?[\d]*)\s*"                                  # numeric value
        r"([a-zA-Z\/\%µgLUd]{1,12})?\s*"                     # unit
        r"([\d]+\.?[\d]*\s*[-–]\s*[\d]+\.?[\d]*)?"           # reference range
    )

    # Pattern B: colon-separated  "Test Name : value unit"
    pattern_colon = re.compile(
        r"^([A-Za-z][A-Za-z\s\(\)\/\-\.]{2,40})\s*[:\-]\s*"
        r"([\d]+\.?[\d]*)\s*"
        r"([a-zA-Z\/\%µgLUd]{1,12})?"
    )

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5 or SKIP_PATTERNS.match(line):
            continue

        m = pattern_cols.match(line) or pattern_colon.match(line)
        if m:
            test = _clean_test_name(m.group(1))
            if len(test) < 3:
                continue
            results.append({
                "test": test,
                "value": m.group(2) if m.lastindex >= 2 else "N/A",
                "unit": m.group(3) if m.lastindex >= 3 and m.group(3) else "",
                "reference": m.group(4) if m.lastindex >= 4 and m.group(4) else "",
                "status": "",
            })

    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(lab_values: list[dict]) -> list[dict]:
    """Remove duplicate test entries, keeping the one with more data."""
    seen = {}
    for item in lab_values:
        key = item["test"].lower().strip()
        if key not in seen:
            seen[key] = item
        else:
            # Keep whichever has more non-empty fields
            existing_score = sum(1 for v in seen[key].values() if v)
            new_score = sum(1 for v in item.values() if v)
            if new_score > existing_score:
                seen[key] = item
    return list(seen.values())
