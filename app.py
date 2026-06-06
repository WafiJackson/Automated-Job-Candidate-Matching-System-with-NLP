import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from modules.ocr_engine import extract_text_from_image

st.set_page_config(page_title="Sistem Rekrutmen Cerdas", layout="wide")

st.title("Sistem Document Understanding & Job Matchmaking")
st.markdown("Mengekstrak informasi CV menggunakan OCR & NER, lalu mencocokkannya dengan kriteria loker.")

# ==========================================
# 1. MEMUAT MODEL AI (Di-cache agar cepat)
# ==========================================
@st.cache_resource
def load_models():
    # Model NER milikmu hasil download dari Colab
    ner_model_path = "./models/indobert-ner-rekrutmen-final"
    ner_pipe = pipeline("ner", model=ner_model_path, aggregation_strategy="simple")
    
    # Model Embedding untuk revisi dosen (menerjemahkan teks ke vektor)
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return ner_pipe, embedder

with st.spinner("Memuat model AI..."):
    ner_pipe, embedder = load_models()

# ==========================================
# 2. ANTARMUKA PENGGUNA (UI)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.header("1. Kriteria Lowongan")
    job_desc = st.text_area(
        "Masukkan kriteria yang dicari (Skill/Pengalaman):", 
        "Mencari programmer yang menguasai Python, Machine Learning, dan React."
    )

with col2:
    st.header("2. Upload CV Pelamar")
    uploaded_file = st.file_uploader("Upload hasil scan CV (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ==========================================
# 3. PIPELINE EKSEKUSI UTAMA
# ==========================================
if uploaded_file and st.button("Proses CV & Hitung Kecocokan"):
    # Simpan gambar sementara
    temp_image_path = "temp_cv.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # FASE 1: OCR
    with st.spinner("1/3 Mengekstrak teks dari gambar (OCR)..."):
        raw_text = extract_text_from_image(temp_image_path)

    # FASE 2: NER (Mengekstrak hanya skill/pengalaman)
    with st.spinner("2/3 Menganalisis entitas penting (NER)..."):
        entities = ner_pipe(raw_text)
        
        # Filter: Hanya ambil kata yang dilabeli sebagai KETERAMPILAN oleh modelmu
        extracted_skills = [ent['word'] for ent in entities if 'KETERAMPILAN' in ent['entity_group'].upper()]
        cv_summary = " ".join(extracted_skills)

    # FASE 3: EMBEDDING & PENCOCOKAN (Sesuai Revisi)
    with st.spinner("3/3 Menghitung skor kecocokan (Embedding)..."):
        if not cv_summary:
            st.error("Model NER tidak menemukan entitas keterampilan pada CV ini.")
        else:
            # Ubah teks ke vektor
            job_embedding = embedder.encode(job_desc, convert_to_tensor=True)
            cv_embedding = embedder.encode(cv_summary, convert_to_tensor=True)

            # Hitung Cosine Similarity
            cosine_scores = util.cos_sim(cv_embedding, job_embedding)
            match_percentage = cosine_scores[0][0].item() * 100

            # --- TAMPILKAN HASIL ---
            st.success("Pemrosesan Selesai!")
            st.metric(label="Tingkat Kecocokan (Similarity Score)", value=f"{match_percentage:.2f}%")
            
            st.write("### Detail Ekstraksi NER:")
            st.write(f"**Keterampilan/Entitas yang Ditemukan:** {', '.join(extracted_skills)}")

            with st.expander("Lihat Teks Mentah Hasil OCR"):
                st.write(raw_text)