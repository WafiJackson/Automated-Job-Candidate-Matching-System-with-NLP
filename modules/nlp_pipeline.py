import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

import os

def load_ai_models():
    """Memuat model AI (NER dan SentenceTransformer)."""
    wikiann_path = "./models/final_wikiann_model"
    rekrutmen_path = "./models/indobert-ner-rekrutmen-final"
    
    if not os.path.exists(wikiann_path) or not os.path.exists(rekrutmen_path):
        raise FileNotFoundError("Model AI tidak ditemukan! Harap jalankan 'fine_tuning_ner.ipynb' terlebih dahulu dan simpan model di folder 'models/'.")
        
    wikiann_pipe = pipeline("ner", model=wikiann_path, aggregation_strategy="simple")
    rekrutmen_pipe = pipeline("ner", model=rekrutmen_path, aggregation_strategy="simple")
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    return wikiann_pipe, rekrutmen_pipe, embedder

def clean_entity_word(w: str) -> str:
    w = w.replace(" ##", "").replace("##", "")
    return " ".join(w.split())

def process_extracted_entities(ents, label_filter, is_wiki=False):
    # Stopwords & noisy words yang sering disalahpahami oleh model sebagai keterampilan
    blacklist_skills = {
        'aplikasi', 'web', 'sistem', 'berbasis', 'pengembangan', 'untuk', 'perusahaan', 
        'bahasa', 'pemrograman', 'waktu', 'it', 'organisasi', 'analisis', 'kelompok', 
        'masa', 'tantangan', 'efek', 'masalah', 'perkembangan', 'akhir', '00', 'dan', 
        'yang', 'atau', 'dengan', 'dalam', 'dari', 'pada', 'adalah', 'sebagai', 'oleh', 
        'saya', 'kami', 'mereka', 'dia', 'kamu', 'anda', 'kita', 'tugas', 'proyek', 
        'menggunakan', 'membuat', 'melakukan', 'kerja', 'tim', 'komunikasi', 'efektif', 
        'manajemen', 'studi', 'kasus', 'sekolah', 'kuliah', 'jurusan', 'informatika', 
        'indonesia', 'buku', 'serta', 'dapat', 'bisa', 'akan', 'telah', 'sudah', 'belum',
        'hard', 'soft', 'skills', 'skill', 'software', 'berbasis web', 'web development',
        'development', 'pemrogram', 'pemrogrammer', 'nat', 'ent', 'dar', 'jal', 'ac', 'id',
        'informasi', 'internal', 'informasi internal', 'database', 'studio', 'studio code'
    }
    
    valid_short_skills = {'go', 'c', 'r', 'js', 'c#', 'c++', 'db', 'qt', 'ip', 'ui', 'ux', 'php', 'sql'}

    extracted = []
    extracted_lower = set()
    for ent in ents:
        group = ent['entity_group'].upper()
        word = ent['word'].strip()
        word = clean_entity_word(word)
        
        # Filter karakter/simbol sampah
        if not word or word in [":", "-", ",", ".", "/", "\\", "|", "•", "©", "®"] or len(word) <= 1:
            continue
            
        word_lower = word.lower()
        if is_wiki:
            is_match = False
            if label_filter == 'PER' and (any(x in group for x in ['LABEL_1', 'LABEL_2']) or 'PER' in group):
                is_match = True
            elif label_filter == 'ORG' and (any(x in group for x in ['LABEL_3', 'LABEL_4']) or 'ORG' in group):
                is_match = True
            elif label_filter == 'LOC' and (any(x in group for x in ['LABEL_5', 'LABEL_6']) or 'LOC' in group):
                is_match = True
                
            if is_match and word_lower not in extracted_lower:
                if word_lower not in ['universitas', 'sekolah', 'institut', 'lulusan', 'jurusan', 'fakultas']:
                    extracted.append(word)
                    extracted_lower.add(word_lower)
        else:
            if label_filter in group:
                # Pembersihan khusus untuk kategori keterampilan
                if label_filter == 'KETERAMPILAN':
                    # Cek blacklist
                    if word_lower in blacklist_skills:
                        continue
                    # Cek panjang karakter minimum (<= 2) kecuali terdaftar sebagai skill valid
                    if len(word) <= 2 and word_lower not in valid_short_skills:
                        continue
                
                if word_lower not in extracted_lower:
                    extracted.append(word)
                    extracted_lower.add(word_lower)
    return extracted

def run_ner_extraction(raw_text, wikiann_pipe, rekrutmen_pipe):
    """Menjalankan ekstraksi NER pada teks mentah dan membersihkannya."""
    entities_wiki = wikiann_pipe(raw_text)
    entities_rek = rekrutmen_pipe(raw_text)

    # Ekstrak data profil dari model WikiANN
    names = process_extracted_entities(entities_wiki, 'PER', is_wiki=True)
    institutions = process_extracted_entities(entities_wiki, 'ORG', is_wiki=True)
    locations = process_extracted_entities(entities_wiki, 'LOC', is_wiki=True)

    # Ekstrak data karir dari model Rekrutmen
    extracted_skills = process_extracted_entities(entities_rek, 'KETERAMPILAN', is_wiki=False)
    extracted_jobs = process_extracted_entities(entities_rek, 'JABATAN', is_wiki=False)

    # Kamus Keterampilan Umum (Lexicon Match)
    common_skills_lexicon = [
        'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'php', 'ruby', 'swift', 'kotlin', 'golang', 'rust', 'c', 'sql', 'r', 'dart',
        'html', 'css', 'reactjs', 'react', 'vuejs', 'vue', 'angular', 'laravel', 'django', 'flask', 'spring', 'express', 'nodejs', 'nextjs', 'bootstrap', 'tailwind',
        'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'sqlite', 'mariadb', 'oracle', 'firebase',
        'git', 'github', 'gitlab', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'postman', 'vscode', 'visual studio code', 'figma',
        'machine learning', 'deep learning', 'nlp', 'natural language processing', 'ai', 'artificial intelligence', 'data analysis', 'data science', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn'
    ]

    raw_text_lower = raw_text.lower()
    extracted_skills_lower = {s.lower() for s in extracted_skills}
    
    for skill in common_skills_lexicon:
        if '+' in skill or '#' in skill:
            if skill in raw_text_lower:
                idx = raw_text_lower.find(skill)
                orig_skill = raw_text[idx:idx+len(skill)].strip()
                if orig_skill and orig_skill.lower() not in extracted_skills_lower:
                    extracted_skills.append(orig_skill)
                    extracted_skills_lower.add(orig_skill.lower())
        else:
            pattern = rf'\b{re.escape(skill)}\b'
            matches = list(re.finditer(pattern, raw_text_lower))
            if matches:
                idx = matches[0].start()
                orig_skill = raw_text[idx:idx+len(skill)].strip()
                
                lower_name = orig_skill.lower()
                if lower_name in ['javascript', 'typescript', 'mongodb', 'postgresql', 'reactjs', 'vuejs', 'nextjs', 'vscode', 'postman']:
                    cap_map = {
                        'javascript': 'JavaScript', 'typescript': 'TypeScript', 'mongodb': 'MongoDB',
                        'postgresql': 'PostgreSQL', 'reactjs': 'ReactJS', 'vuejs': 'Vue.js',
                        'nextjs': 'Next.js', 'vscode': 'VS Code', 'postman': 'Postman'
                    }
                    orig_skill = cap_map.get(lower_name, orig_skill)
                elif len(orig_skill) <= 4:
                    orig_skill = orig_skill.upper()
                else:
                    orig_skill = orig_skill.capitalize()
                    
                if orig_skill.lower() in extracted_skills_lower:
                    for idx_enum, s in enumerate(extracted_skills):
                        if s.lower() == orig_skill.lower():
                            extracted_skills.pop(idx_enum)
                            break
                
                extracted_skills.append(orig_skill)
                extracted_skills_lower.add(orig_skill.lower())

    entities = entities_wiki + entities_rek
    return names, institutions, locations, extracted_skills, extracted_jobs, entities

def extract_contact_info(raw_text: str) -> dict:
    """
    Mengekstrak informasi kontak (email dan nomor telepon) dari teks CV menggunakan Regex.
    
    Returns:
        dict: {'email': str|None, 'phone': str|None}
    """
    # Pola email umum
    email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, raw_text)
    email = email_match.group(0) if email_match else None

    # Pola nomor telepon Indonesia (+62 atau 08xx) dengan berbagai format pemisah
    phone_pattern = r'(?:\+62|\(\+62\)|0)[\s.\-]?(?:8[0-9]{2}|21|22|31|274|361)[\s.\-]?[0-9]{3,4}[\s.\-]?[0-9]{3,5}'
    phone_match = re.search(phone_pattern, raw_text)
    phone = phone_match.group(0).strip() if phone_match else None

    return {'email': email, 'phone': phone}

def calculate_similarity(extracted_jobs, extracted_skills, job_desc, raw_text, embedder):
    """
    Menghitung skor kecocokan antara CV dan deskripsi pekerjaan menggunakan Hybrid Scoring.
    
    Skor Akhir = (60% x Cosine Similarity) + (40% x Keyword Match)
    - Cosine Similarity: Mengukur kedekatan semantik secara keseluruhan.
    - Keyword Match: Mengukur berapa persen kata kunci dari job desc yang benar-benar ada di CV.
    """
    # --- Komponen 1: Cosine Similarity (60%) ---
    cv_structured_text = f"Jabatan: {', '.join(extracted_jobs)}. Keterampilan: {', '.join(extracted_skills)}"
    if not extracted_jobs and not extracted_skills:
        cv_structured_text = raw_text.strip()[:500]

    job_embedding = embedder.encode(job_desc, convert_to_tensor=True)
    cv_embedding = embedder.encode(cv_structured_text, convert_to_tensor=True)
    cosine_score = util.cos_sim(cv_embedding, job_embedding)[0][0].item() * 100

    # --- Komponen 2: Keyword Match / Jaccard (40%) ---
    # Ekstrak kata bermakna dari job description (filter stopwords singkat)
    job_words = set(re.findall(r'\b[a-zA-Z0-9+#]{2,}\b', job_desc.lower()))
    cv_words = set(re.findall(r'\b[a-zA-Z0-9+#]{2,}\b', ' '.join(extracted_skills + extracted_jobs).lower()))

    if job_words:
        keyword_score = (len(job_words & cv_words) / len(job_words)) * 100
    else:
        keyword_score = 0.0

    # --- Skor Akhir Hybrid ---
    hybrid_score = (0.6 * cosine_score) + (0.4 * keyword_score)

    return round(hybrid_score, 2), round(cosine_score, 2), round(keyword_score, 2)
