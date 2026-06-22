import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

import os

def load_ai_models():
    """Memuat model AI (NER dan SentenceTransformer)."""
    wikiann_path = "./models/final_wikiann_model"
    rekrutmen_path = "./models/indobert-ner-rekrutmen-final"
    
    if not os.path.exists(wikiann_path) or not os.path.exists(rekrutmen_path):
        raise FileNotFoundError("Model AI tidak ditemukan! Harap jalankan 'fine_tuning_ner.ipynb' terlebih dahulu dan simpan model di folder 'models/'.")
        
    wikiann_pipe = pipeline("ner", model=wikiann_path, aggregation_strategy="first")
    rekrutmen_pipe = pipeline("ner", model=rekrutmen_path, aggregation_strategy="first")
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    return wikiann_pipe, rekrutmen_pipe, embedder, cross_encoder

COMMON_SKILLS_LEXICON = [
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'php', 'ruby', 'swift', 'kotlin', 'golang', 'rust', 'c', 'sql', 'r', 'dart',
    'html', 'css', 'reactjs', 'react', 'vuejs', 'vue', 'angular', 'laravel', 'django', 'flask', 'spring', 'express', 'nodejs', 'nextjs', 'bootstrap', 'tailwind',
    'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'sqlite', 'mariadb', 'oracle', 'firebase',
    'git', 'github', 'gitlab', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'postman', 'vscode', 'visual studio code', 'figma',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'ai', 'artificial intelligence', 'data analysis', 'data science', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn'
]

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
        'informasi', 'internal', 'informasi internal', 'database', 'studio', 'studio code',
        'merancang', 'solusi', 'interpretasi', 'bisnis', 'keputusan', 'melalui', 'dasar',
        'strategi', 'pemasaran', 'sampai', 'sekarang', 'cohort', 'mengim', 'ro', 'pt', 'hp',
        'abc', 'def', 'ple', 'data', 'sci', 'mengembangkan', 'membantu', 'mengambil', 'menyusun',
        'meningkatkan', 'berkolaborasi', 'melakukan', 'mengidentifikasi', 'harga', 'hingga',
        # English Stopwords & Noise
        'and', 'the', 'with', 'for', 'you', 'are', 'is', 'from', 'have', 'has', 'was', 'this', 
        'that', 'which', 'using', 'new', 'team', 'work', 'experience', 'years', 'master', 
        'advanced', 'knowledge', 'proven', 'track', 'record', 'of', 'at', 'in', 'on', 'by',
        'to', 'as', 'an', 'a', 'or', 'not', 'be', 'it', 'can', 'will', 'all', 'any', 'other',
        'some', 'such', 'more', 'most', 'very', 'too', 'also', 'only', 'even', 'just', 'then',
        'than', 'so', 'if', 'but', 'because', 'while', 'where', 'when', 'how', 'why', 'what',
        'who', 'whom', 'whose', 'which', 'whether', 'whose', 'whoever', 'whomever', 'whatever',
        'whichever', 'whenever', 'wherever', 'however', 'whyever', 'help', 'hancv', 'mar', 'st'
    }
    
    valid_short_skills = {'go', 'c', 'r', 'js', 'c#', 'c++', 'db', 'qt', 'ip', 'ui', 'ux', 'php', 'sql'}

    # Threshold confidence score — hanya terima entitas yang modelnya yakin
    # JABATAN: minimal 0.60, lainnya (NAMA, ORG, LOC): minimal 0.50
    SCORE_THRESHOLD_JABATAN = 0.60
    SCORE_THRESHOLD_OTHER   = 0.50

    extracted = []
    extracted_lower = set()
    for ent in ents:
        group = ent['entity_group'].upper()
        score = ent.get('score', 1.0)
        word = ent['word'].strip()
        word = clean_entity_word(word)
        
        # Filter karakter/simbol sampah & subword fragment (##...)
        if not word or word in [":", "-", ",", ".", "/", "\\", "|", "•", "©", "®"] or len(word) <= 1:
            continue
        if word.startswith('#'):
            continue
            
        word_lower = word.lower()
        if is_wiki:
            # WikiANN — minimal confidence
            if score < SCORE_THRESHOLD_OTHER or word_lower in blacklist_skills:
                continue
            is_match = False
            if label_filter == 'PER' and (any(x in group for x in ['LABEL_1', 'LABEL_2']) or 'PER' in group):
                is_match = True
            elif label_filter == 'ORG' and (any(x in group for x in ['LABEL_3', 'LABEL_4']) or 'ORG' in group):
                is_match = True
            elif label_filter == 'LOC' and (any(x in group for x in ['LABEL_5', 'LABEL_6']) or 'LOC' in group):
                is_match = True
                
            if is_match and word_lower not in extracted_lower:
                if word_lower not in ['universitas', 'sekolah', 'institut', 'lulusan', 'jurusan', 'fakultas']:
                    if len(word.split()) <= 7:  # Mencegah halusinasi WikiANN membaca 1 paragraf sebagai nama
                        extracted.append(word)
                        extracted_lower.add(word_lower)
        else:
            if label_filter in group:
                # Threshold confidence per kategori
                threshold = SCORE_THRESHOLD_JABATAN if label_filter == 'JABATAN' else SCORE_THRESHOLD_OTHER
                if score < threshold:
                    continue
                
                if label_filter == 'JABATAN':
                    if word_lower in blacklist_skills or word.isnumeric():
                        continue
                    if len(word) <= 3:
                        continue
                
                if word_lower not in extracted_lower:
                    if len(word.split()) <= 7:  # Mencegah halusinasi jabatan yang terlalu panjang
                        extracted.append(word)
                        extracted_lower.add(word_lower)
    return extracted

def segment_cv(raw_text):
    """
    Memecah teks CV berdasarkan header yang umum (Pendidikan, Pengalaman, Keahlian).
    """
    sections = {
        'PROFILE': '',
        'EDUCATION': '',
        'EXPERIENCE': '',
        'SKILLS': '',
        'UNKNOWN': ''
    }
    
    # Pola regex untuk mendeteksi header (huruf kecil)
    patterns = {
        'EDUCATION': r'^(?:pendidikan|education|riwayat akademis|latar belakang pendidikan|edukasi)\b',
        'EXPERIENCE': r'^(?:pengalaman|pengalaman kerja|work experience|riwayat pekerjaan|employment history|experience|karir)\b',
        'SKILLS': r'^(?:keahlian|keterampilan|skills?|kompetensi|kemampuan|competencies)\b'
    }
    
    current_section = 'PROFILE'
    lines = raw_text.split('\n')
    
    for line in lines:
        cleaned_line = line.strip().lower()
        # Bersihkan karakter spesial di awal baris (misal bullet point '-')
        header_check = re.sub(r'^[^a-z0-9]+', '', cleaned_line).strip()
        
        # Header umumnya adalah kalimat pendek (maksimal 4 kata)
        if len(header_check.split()) <= 4 and header_check:
            matched = False
            for sec, pattern in patterns.items():
                if re.match(pattern, header_check):
                    current_section = sec
                    matched = True
                    break
        
        sections[current_section] += line + '\n'
        
    return sections

def run_ner_extraction(raw_text, wikiann_pipe, rekrutmen_pipe):
    """Menjalankan ekstraksi NER secara optimal."""
    
    sections = segment_cv(raw_text)
    
    # Cek apakah segmentasi berhasil (ada bagian selain PROFILE yang terisi)
    is_segmented = any(len(sections[sec].strip()) > 0 for sec in ['EDUCATION', 'EXPERIENCE', 'SKILLS'])
    
    entities_wiki = []
    entities_rek = []
    
    if is_segmented:
        # OPTIMASI 1 (Segmented): WikiANN hanya untuk Profile dan Education
        wiki_text = sections['PROFILE'] + "\n" + sections['EDUCATION']
        wiki_words = wiki_text.split()[:300]
        intro_text = " ".join(wiki_words)
        entities_wiki = wikiann_pipe(intro_text) if intro_text.strip() else []
        
        # OPTIMASI 2 (Segmented): Rekrutmen hanya untuk Pengalaman dan Profile
        rek_text = sections['EXPERIENCE'] + "\n" + sections['PROFILE']
        ner_words = rek_text.split()[:450]
        chunk_size = 225
        chunks = [" ".join(ner_words[i:i + chunk_size]) for i in range(0, len(ner_words), chunk_size)]
        for chunk in chunks:
            if chunk.strip():
                entities_rek.extend(rekrutmen_pipe(chunk))
    else:
        # FALLBACK: Logika 1D Memanjang jika CV tidak beraturan/tidak ada header
        words = raw_text.split()
        intro_text = " ".join(words[:200])
        entities_wiki = wikiann_pipe(intro_text) if intro_text.strip() else []

        ner_words = words[:450]
        chunk_size = 225
        chunks = [" ".join(ner_words[i:i + chunk_size]) for i in range(0, len(ner_words), chunk_size)]
        for chunk in chunks:
            if chunk.strip():
                entities_rek.extend(rekrutmen_pipe(chunk))
    
    # -----------------------------------------------------------------------
    # KETERAMPILAN: Gunakan HANYA Lexicon (Regex), BUKAN NER
    # IndoBERT-Rekrutmen cenderung menandai kata kerja Indonesia sebagai skill.
    # Lexicon jauh lebih presisi untuk deteksi skill IT/teknis.
    # -----------------------------------------------------------------------

    # Ekstrak data profil dari model WikiANN (Batasi maksimal 3 agar tidak bocor)
    names = process_extracted_entities(entities_wiki, 'PER', is_wiki=True)[:3]
    institutions = process_extracted_entities(entities_wiki, 'ORG', is_wiki=True)[:3]
    locations = process_extracted_entities(entities_wiki, 'LOC', is_wiki=True)[:3]

    # Ekstrak data karir dari model Rekrutmen
    extracted_jobs = process_extracted_entities(entities_rek, 'JABATAN', is_wiki=False)
    extracted_skills = []  # akan diisi murni oleh Lexicon Regex di bawah

    # Kamus Keterampilan Umum (Lexicon Match)
    common_skills_lexicon = COMMON_SKILLS_LEXICON

    # Jika tersedimentasi, cukup cari skill di bagian relevan
    skill_text = (sections['SKILLS'] + "\n" + sections['EXPERIENCE'] + "\n" + sections['PROFILE']) if is_segmented else raw_text
    raw_text_lower = skill_text.lower()
    extracted_skills_lower = {s.lower() for s in extracted_skills}
    
    for skill in common_skills_lexicon:
        if '+' in skill or '#' in skill:
            if skill in raw_text_lower:
                idx = raw_text_lower.find(skill)
                orig_skill = skill_text[idx:idx+len(skill)].strip()
                if orig_skill and orig_skill.lower() not in extracted_skills_lower:
                    extracted_skills.append(orig_skill)
                    extracted_skills_lower.add(orig_skill.lower())
        else:
            pattern = rf'\b{re.escape(skill)}\b'
            matches = list(re.finditer(pattern, raw_text_lower))
            if matches:
                idx = matches[0].start()
                orig_skill = skill_text[idx:idx+len(skill)].strip()
                
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

def calculate_similarity(extracted_jobs, extracted_skills, job_desc, raw_text, embedder, cross_encoder):
    """
    Menghitung skor kecocokan menggunakan Bi-Encoder (Cosine Similarity) dan Cross-Encoder.
    """
    # 1. Bi-Encoder (Cosine Similarity) 
    # Kita berikan Teks CV yang kaya (Rich Context) agar vektor Cosine bisa menangkap relasi kosakata yang luas
    rich_cv_text = f"{raw_text.strip()[:1000]} Keterampilan Utama: {', '.join(extracted_skills)} Jabatan Terdeteksi: {', '.join(extracted_jobs)}"
    
    job_embedding = embedder.encode(job_desc, convert_to_tensor=True)
    cv_embedding = embedder.encode(rich_cv_text, convert_to_tensor=True)
    cosine_score = util.cos_sim(cv_embedding, job_embedding)[0][0].item() * 100

    # 2. Cross-Encoder
    # MMARCO butuh kalimat alami, bukan teks mentah
    jobs_str = ", ".join(extracted_jobs) if extracted_jobs else "profesional IT"
    skills_str = ", ".join(extracted_skills) if extracted_skills else "kemampuan dasar"
    cv_pure_text = f"Pelamar ini memiliki keahlian di bidang {skills_str}. Ia pernah menjabat sebagai {jobs_str}."
    
    raw_cross_score = cross_encoder.predict([(job_desc, cv_pure_text)])[0]
    prob_score = torch.sigmoid(torch.tensor(raw_cross_score)).item() * 100

    # 3. Exact Keyword Matching (HRD Logic)
    # Ini sangat penting! Jika CV punya skill yang sama persis dengan loker, berikan bonus besar!
    import re
    job_desc_lower = job_desc.lower()
    match_count = 0
    for skill in extracted_skills:
        # Gunakan regex word boundary agar huruf "R" tidak terdeteksi di dalam kata "daRi"
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, job_desc_lower):
            match_count += 1
            
    # Setiap skill yang cocok persis mendapat bonus 8%! (Maksimal bonus 40%)
    keyword_bonus = min(40.0, match_count * 8.0) 

    # 4. Final Blended Score (Sesuai Permintaan User: Harus bisa mencapai 90%+)
    # Kita prioritaskan Cosine (stabil) + Bonus Skill Mutlak. Cross-Encoder hanya pelengkap 10%.
    adjusted_score = (cosine_score * 0.9) + (prob_score * 0.1) + keyword_bonus
    
    # Cap maksimal 99%
    if adjusted_score > 99.5:
        adjusted_score = 99.5

    return round(adjusted_score, 2), round(cosine_score, 2), match_count
