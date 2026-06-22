from transformers import pipeline
import sys
sys.path.append("/home/azani/Kuliah/semester_6/nlp/UAS/")

text = """Alba Christo
DATA SCIENTIST | PYTHON
Makassar, Indonesia | HP: +62-822 123456789 | Email: albachristo@email.com
RINGKASAN
Data scientist andal dengan pengalaman 2 tahun. Memiliki pengetahuan mendalam terkait
machine learning dan analisis statistik. Terbukti mampu merancang dan mengimplementasikan
solusi berbasis analisis dan interpretasi data untuk mengembangkan bisnis. Siapo membantu PT
Datawira mengambil keputusan lebih baik melalui data.
Junior Data Scientist
Juli 2022 — e Menganalisis data sebagai dasar tim marketing PT ABC menyusun
sekarang strategi pemasaran dan meningkatkan ROI sampai 15%
< e Berkolaborasi dengan product manager PT DEF untuk melakukan
= PT Data Maju, analisis cohort yang mengidentifikasi pbengurangan harga hingga
a Makassar 10%"""

wikiann_pipe = pipeline("ner", model="/home/azani/Kuliah/semester_6/nlp/UAS/models/final_wikiann_model", aggregation_strategy="simple")
rekrutmen_pipe = pipeline("ner", model="/home/azani/Kuliah/semester_6/nlp/UAS/models/indobert-ner-rekrutmen-final", aggregation_strategy="simple")

print("WIKIANN:")
for e in wikiann_pipe(text[:200]):
    print(e)

print("\nREKRUTMEN:")
for e in rekrutmen_pipe(text):
    print(e)
