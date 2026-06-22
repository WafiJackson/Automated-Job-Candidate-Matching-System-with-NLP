from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
job_desc = "Mencari Data Scientist dengan kemampuan kuat di bidang Python, Machine Learning, SQL, dan Data Analysis. Pengalaman dengan model AI dan visualisasi data sangat disukai."
cv_structured_text = "Keterampilan pelamar: Python, Machine learning, Analisis statistik, SQL, Excel, SAS. Pengalaman dan Jabatan: Data Scientist, Junior Data Scientist."

job_emb = embedder.encode(job_desc, convert_to_tensor=True)
cv_emb = embedder.encode(cv_structured_text, convert_to_tensor=True)
print("New Score:", util.cos_sim(cv_emb, job_emb)[0][0].item() * 100)
