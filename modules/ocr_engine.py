import cv2
import pytesseract

# Lazy initialization - PaddleOCR hanya dimuat saat pertama kali dibutuhkan
_ocr_instance = None

def _get_ocr():
    """Mengembalikan instance PaddleOCR, dibuat hanya sekali (lazy init)."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='id', enable_mkldnn=False)
    return _ocr_instance

def extract_text_from_image(image_path):
    # 1. Ekstraksi via Tesseract OCR v5
    img = cv2.imread(image_path)
    text_tesseract = pytesseract.image_to_string(img, lang='ind+eng')

    # 2. Ekstraksi via PaddleOCR (dimuat saat pertama kali dipanggil)
    ocr_paddle = _get_ocr()
    result = ocr_paddle.ocr(image_path)
    text_paddle = ""
    if result and result[0]:
        text_paddle = " ".join([line[1][0] for line in result[0]])

    # Strategi Ensemble Sederhana:
    # Kita gabungkan teksnya, biarkan model NER yang menyaring mana teks yang relevan
    combined_text = text_tesseract + "\n" + text_paddle
    return combined_text