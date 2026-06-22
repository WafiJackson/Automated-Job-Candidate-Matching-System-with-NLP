import sys

with open("app.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False

for i, line in enumerate(lines):
    if line.strip() == '# PHASE 1: OCR':
        new_lines.append("    # PHASE 1: OCR / Text Extraction\n")
        new_lines.append("    with st.spinner(\"🔍 Mengekstrak teks dari dokumen...\"):\n")
        new_lines.append("        if uploaded_file.name.lower().endswith('.pdf'):\n")
        new_lines.append("            raw_text, extraction_method = process_pdf(temp_image_path)\n")
        new_lines.append("            st.info(f\"💡 Metode Ekstraksi PDF: {extraction_method}\")\n")
        new_lines.append("        else:\n")
        new_lines.append("            raw_text = extract_text_from_image(temp_image_path)\n")
        skip = True
    elif skip and line.strip() == '# Progress pipeline - NER active':
        skip = False
        new_lines.append(line)
    elif line.strip() == '# PHASE 2: NER':
        new_lines.append("    # PHASE 2: NER\n")
        new_lines.append("    with st.spinner(\"🏷️ Menganalisis entitas penting (Dual NER)...\"):\n")
        new_lines.append("        names, institutions, locations, extracted_skills, extracted_jobs, entities = run_ner_extraction(raw_text, wikiann_pipe, rekrutmen_pipe)\n")
        skip = True
    elif skip and line.strip() == '# Progress pipeline - Matching active':
        skip = False
        new_lines.append(line)
    elif line.strip() == '# PHASE 3: Embedding & Matching':
        new_lines.append("    # PHASE 3: Embedding & Matching\n")
        new_lines.append("    with st.spinner(\"🧬 Menghitung skor kecocokan (Embedding)...\"):\n")
        new_lines.append("        if not raw_text.strip():\n")
        new_lines.append("            st.error(\"⚠️ Teks dokumen kosong atau tidak terbaca. Silakan unggah dokumen yang lebih jelas.\")\n")
        new_lines.append("        else:\n")
        new_lines.append("            match_percentage = calculate_similarity(extracted_jobs, extracted_skills, job_desc, raw_text, embedder)\n")
        skip = True
    elif skip and line.strip() == '# Final pipeline - all done':
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open("app.py", "w") as f:
    f.writelines(new_lines)
