import fitz  # PyMuPDF
from PIL import Image
import io
import os

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except Exception:
    _TESSERACT_AVAILABLE = False

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Identifies file type and extracts raw text.
    Supports PDF and common Image formats (via OCR).
    """
    ext = filename.split(".")[-1].lower()
    
    if ext == "pdf":
        return _extract_from_pdf(file_content)
    elif ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
        return _extract_from_image(file_content)
    else:
        # For text files or unknown formats, try decoding as UTF-8
        try:
            return file_content.decode("utf-8")
        except:
            return ""

def _extract_from_pdf(content: bytes) -> str:
    """Extracts text from PDF. Falls back to OCR if page is scanned."""
    text = ""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            page_text = page.get_text()
            # If a page has very little text, it might be a scanned image
            if len(page_text.strip()) < 10 and _TESSERACT_AVAILABLE:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                try:
                    page_text = pytesseract.image_to_string(img)
                except Exception:
                    page_text = ""
            text += page_text + "\n"
        doc.close()
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
    return text

def _extract_from_image(content: bytes) -> str:
    """Perform OCR on image bytes."""
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        img = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
