import cv2
import pytesseract

def extract_text_from_image(image_path):
    """
    Ekstrak teks dari gambar menggunakan Tesseract OCR.
    Ini sangat cepat, offline, dan bebas dari masalah download model server.
    """
    try:
        # Baca gambar menggunakan OpenCV
        img = cv2.imread(image_path)
        
        # Ekstraksi Teks via Tesseract OCR v5
        # Menggunakan bahasa Indonesia (ind) dan English (eng)
        text_tesseract = pytesseract.image_to_string(img, lang='ind+eng')
        
        return text_tesseract
    except Exception as e:
        print(f"Error pada OCR Tesseract: {e}")
        return ""