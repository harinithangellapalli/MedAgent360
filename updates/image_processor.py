"""
MedAgent 360 · Module B · Step B1
Image Preprocessor for Prescription OCR
Pipeline: grayscale → denoise → binarize → deskew → enhance contrast
Handles both printed and handwritten prescriptions.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from scripts.logger import get_logger

logger = get_logger("prescription.image_processor")

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def preprocess_image(image_path: str, output_path: str | None = None) -> tuple[str, np.ndarray]:
    """
    Full preprocessing pipeline for prescription images.

    Steps:
    1. Load & validate image
    2. Resize to optimal OCR resolution (300 DPI equivalent)
    3. Convert to grayscale
    4. Denoise
    5. Adaptive binarization (handles uneven lighting)
    6. Deskew (straighten tilted photos)
    7. Morphological cleanup

    Returns:
        (output_path, processed_numpy_array)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {path.suffix}. Use: {SUPPORTED_FORMATS}")

    logger.info(f"🖼️  Preprocessing: {path.name}")

    # Step 1: Load
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = img.shape[:2]
    logger.info(f"  Original size: {original_w}x{original_h}")

    # Step 2: Resize for optimal OCR (target ~2400px wide)
    img = _resize_for_ocr(img, target_width=2400)

    # Step 3: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 5: Adaptive thresholding (handles shadows, uneven lighting)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )

    # Step 6: Deskew
    deskewed = _deskew(binary)

    # Step 7: Morphological cleanup — close small gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(deskewed, cv2.MORPH_CLOSE, kernel)

    # Save output
    if output_path is None:
        output_path = str(path.parent / f"{path.stem}_processed.png")

    cv2.imwrite(output_path, cleaned)
    logger.info(f"  ✅ Saved preprocessed image: {output_path}")

    return output_path, cleaned


def preprocess_for_handwritten(image_path: str, output_path: str | None = None) -> tuple[str, np.ndarray]:
    """
    Enhanced pipeline specifically for handwritten prescriptions.
    Uses stronger contrast enhancement and larger morphological kernels.
    """
    path = Path(image_path)
    logger.info(f"✍️  Handwritten preprocessing: {path.name}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize
    img = _resize_for_ocr(img, target_width=2400)

    # Use PIL for better contrast enhancement
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)

    # Back to OpenCV
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heavier denoising for handwriting
    denoised = cv2.fastNlMeansDenoising(gray, h=15)

    # Otsu's binarization (better for handwriting)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    deskewed = _deskew(binary)

    if output_path is None:
        output_path = str(path.parent / f"{path.stem}_handwritten.png")

    cv2.imwrite(output_path, deskewed)
    return output_path, deskewed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_for_ocr(img: np.ndarray, target_width: int = 2400) -> np.ndarray:
    h, w = img.shape[:2]
    if w < target_width:
        scale = target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


def _deskew(binary_img: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle in scanned/photographed documents."""
    try:
        coords = np.column_stack(np.where(binary_img < 128))  # dark pixels
        if len(coords) < 100:
            return binary_img

        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Only correct if skew is significant
        if abs(angle) < 0.5:
            return binary_img

        logger.info(f"  Deskewing: {angle:.2f}°")
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception as e:
        logger.warning(f"  Deskew failed: {e} — using original")
        return binary_img
