from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
job_desc = "Mencari Data Scientist dengan kemampuan kuat di bidang Python, Machine Learning, SQL, dan Data Analysis. Pengalaman dengan model AI dan visualisasi data sangat disukai."
cv_text = "Jabatan: Data Scientist, Junior Data Scientist. Keterampilan: Python, Machine Learning, SQL, Excel, Analisis Statistik"

job_emb = embedder.encode(job_desc, convert_to_tensor=True)
cv_emb = embedder.encode(cv_text, convert_to_tensor=True)
print("Score:", util.cos_sim(cv_emb, job_emb)[0][0].item() * 100)

cv_raw = "Alba Christo DATA SCIENTIST | PYTHON Makassar, Indonesia | HP: +62-822 123456789 | Email: albachristo@email.com RINGKASAN Data scientist andal dengan pengalaman 2 tahun. Memiliki pengetahuan mendalam terkait machine learning dan analisis statistik."
cv_raw_emb = embedder.encode(cv_raw, convert_to_tensor=True)
print("Raw Score:", util.cos_sim(cv_raw_emb, job_emb)[0][0].item() * 100)
