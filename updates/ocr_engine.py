"""
MedAgent 360 · Module B · Step B2
Tesseract OCR Engine
Handles both printed and handwritten prescription text extraction.
Supports English, Telugu (tel), and Hindi (hin) scripts.
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from scripts.logger import get_logger

logger = get_logger("prescription.ocr_engine")

# Tesseract configs for different prescription types
OCR_CONFIGS = {
    "printed_english": "--oem 3 --psm 6 -l eng",
    "printed_telugu":  "--oem 3 --psm 6 -l tel+eng",
    "printed_hindi":   "--oem 3 --psm 6 -l hin+eng",
    "handwritten":     "--oem 1 --psm 6 -l eng",   # LSTM only, better for handwriting
    "mixed":           "--oem 3 --psm 11 -l eng",  # Sparse text mode
}


def extract_text_from_image(
    image_input,  # path str or numpy array
    script: str = "printed_english",
    return_confidence: bool = False,
) -> dict:
    """
    Run Tesseract OCR on a preprocessed prescription image.

    Args:
        image_input: File path or preprocessed numpy array.
        script: OCR config key from OCR_CONFIGS.
        return_confidence: Whether to return per-word confidence scores.

    Returns:
        {
            "raw_text": str,
            "lines": list[str],
            "confidence": float,  # average confidence
            "word_data": list[dict]  # if return_confidence=True
        }
    """
    # Load image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_input}")
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:  # already grayscale
            pil_img = Image.fromarray(image_input)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        raise TypeError("image_input must be a file path or numpy array")

    config = OCR_CONFIGS.get(script, OCR_CONFIGS["printed_english"])
    logger.info(f"🔤 Running OCR | config={script}")

    # Extract text
    raw_text = pytesseract.image_to_string(pil_img, config=config)
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    # Get confidence data
    try:
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    except Exception:
        avg_confidence = 0.0
        data = {}

    logger.info(f"  Extracted {len(lines)} lines | avg confidence: {avg_confidence:.1f}%")

    result = {
        "raw_text": raw_text.strip(),
        "lines": lines,
        "confidence": round(avg_confidence, 1),
    }

    if return_confidence and data:
        result["word_data"] = [
            {"word": w, "conf": c}
            for w, c in zip(data.get("text", []), data.get("conf", []))
            if w.strip() and str(c).isdigit() and int(c) > 0
        ]

    return result


def auto_detect_and_extract(image_path: str) -> dict:
    """
    Auto-detect prescription type (printed/handwritten/mixed) and
    run the best OCR configuration. Falls back gracefully.
    """
    from prescription.image_processor import preprocess_image, preprocess_for_handwritten

    logger.info(f"🔍 Auto-detecting prescription type: {Path(image_path).name}")

    # Try printed first
    processed_path, processed_arr = preprocess_image(image_path)
    result_printed = extract_text_from_image(processed_arr, script="printed_english")

    # If confidence is low, try handwritten mode
    if result_printed["confidence"] < 50:
        logger.info("  Low confidence — trying handwritten mode")
        hw_path, hw_arr = preprocess_for_handwritten(image_path)
        result_hw = extract_text_from_image(hw_arr, script="handwritten")

        if result_hw["confidence"] > result_printed["confidence"]:
            logger.info(f"  Handwritten mode better: {result_hw['confidence']:.1f}% vs {result_printed['confidence']:.1f}%")
            result_hw["detection_mode"] = "handwritten"
            return result_hw

    result_printed["detection_mode"] = "printed"
    return result_printed
