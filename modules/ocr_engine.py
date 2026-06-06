import cv2
import pytesseract
from paddleocr import PaddleOCR

# Inisialisasi PaddleOCR (Pretrained)
ocr_paddle = PaddleOCR(use_angle_cls=True, lang='id', enable_mkldnn=False)

def extract_text_from_image(image_path):
    # 1. Ekstraksi via Tesseract OCR v5
    img = cv2.imread(image_path)
    text_tesseract = pytesseract.image_to_string(img, lang='ind+eng')

    # 2. Ekstraksi via PaddleOCR
    result = ocr_paddle.ocr(image_path)
    text_paddle = ""
    if result and result[0]:
        text_paddle = " ".join([line[1][0] for line in result[0]])

    # Strategi Ensemble Sederhana: 
    # Kita gabungkan teksnya, biarkan model NER yang menyaring mana teks yang relevan
    combined_text = text_tesseract + "\n" + text_paddle
    return combined_text