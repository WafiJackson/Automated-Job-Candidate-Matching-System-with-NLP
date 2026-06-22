import cv2
import pytesseract

# Lazy initialization - PaddleOCR hanya dimuat saat pertama kali dibutuhkan
_ocr_instance = None

def _get_ocr():
    """Mengembalikan instance PaddleOCR, dibuat hanya sekali (lazy init)."""
    global _ocr_instance
    if _ocr_instance is None:
        import os
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='id', enable_mkldnn=False)
    return _ocr_instance

def extract_text_from_image(image_path):
    # Ekstraksi Utama via PaddleOCR (Lebih baik untuk dokumen terstruktur)
    ocr_paddle = _get_ocr()
    result = ocr_paddle.ocr(image_path)
    text_paddle = ""
    if result and result[0]:
        text_paddle = " ".join([line[1][0] for line in result[0]])

    # Strategi Fallback
    # Jika PaddleOCR berhasil dan teks cukup panjang, kembalikan langsung untuk mencegah duplikasi
    if len(text_paddle.strip()) > 50:
        return text_paddle

    # Ekstraksi Cadangan via Tesseract OCR v5
    img = cv2.imread(image_path)
    text_tesseract = pytesseract.image_to_string(img, lang='ind+eng')

    # Jika PaddleOCR sangat pendek, gabungkan dengan Tesseract (atau gunakan Tesseract saja)
    return text_paddle + "\n" + text_tesseract if text_paddle else text_tesseract