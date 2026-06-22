import fitz  # PyMuPDF
import os
from modules.ocr_engine import extract_text_from_image

def process_pdf(pdf_path, temp_image_path="temp_cv.jpg"):
    """
    Memproses file PDF:
    1. Mencoba mengekstrak teks bawaan secara langsung (sangat cepat dan 100% akurat).
    2. Jika tidak ada teks (PDF hasil scan), ubah halaman pertama menjadi gambar dan lakukan OCR.
    
    Returns:
        tuple: (raw_text, method_used)
    """
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return "", "Failed (Empty PDF)"
            
        page = doc[0] # Ambil halaman pertama saja untuk CV
        text = page.get_text()
        
        # Jika teks cukup panjang, berarti ini PDF text-based
        if len(text.strip()) > 50:
            return text, "Direct Text Extraction (PDF)"
            
        # Jika teks kosong atau sangat pendek, kemungkinan ini PDF hasil scan
        # Render halaman pertama menjadi gambar
        pix = page.get_pixmap(dpi=300) # Resolusi tinggi untuk OCR yang lebih baik
        pix.save(temp_image_path)
        
        # Jalankan OCR pada gambar yang dihasilkan
        extracted_text = extract_text_from_image(temp_image_path)
        return extracted_text, "OCR on Scanned PDF"
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return "", f"Error: {e}"
