import streamlit as st
import requests
import base64
from streamlit_lottie import st_lottie
from modules.nlp_pipeline import load_ai_models, run_ner_extraction, calculate_similarity, extract_contact_info
from modules.ocr_engine import extract_text_from_image
from modules.pdf_handler import process_pdf

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RecruitAI — Smart Talent Matching",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOTTIE HELPERS
# ==========================================
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ==========================================
# MASSIVE CUSTOM CSS
# ==========================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --bg-primary: #0a0615;
        --bg-secondary: #110d21;
        --bg-card: rgba(20, 14, 40, 0.65);
        --bg-card-hover: rgba(30, 20, 60, 0.75);
        --accent-purple: #a855f7;
        --accent-violet: #8b5cf6;
        --accent-pink: #ec4899;
        --accent-blue: #6366f1;
        --accent-cyan: #22d3ee;
        --accent-emerald: #10b981;
        --gradient-main: linear-gradient(135deg, #a855f7, #ec4899, #6366f1);
        --gradient-subtle: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(236,72,153,0.08));
        --gradient-border: linear-gradient(135deg, #a855f7, #ec4899, #6366f1, #22d3ee);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --glass-border: rgba(168, 85, 247, 0.2);
        --shadow-glow: 0 0 30px rgba(168, 85, 247, 0.15);
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-display: 'Poppins', 'Inter', sans-serif;
    }

    /* ===== GLOBAL STYLES ===== */
    .stApp {
        font-family: var(--font-main) !important;
    }
    
    .stApp > header {
        background: transparent !important;
    }

    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* ===== ANIMATED BACKGROUND ===== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(ellipse 80% 50% at 20% 20%, rgba(139, 92, 246, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 60%),
            radial-gradient(ellipse 50% 30% at 50% 50%, rgba(99, 102, 241, 0.06) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
        animation: bgShift 15s ease-in-out infinite alternate;
    }

    @keyframes bgShift {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }

    /* ===== FLOATING PARTICLES ===== */
    .particles-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .particles-bg::before,
    .particles-bg::after {
        content: '';
        position: absolute;
        border-radius: 50%;
        animation: float 20s ease-in-out infinite;
    }

    .particles-bg::before {
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.08) 0%, transparent 70%);
        top: 10%;
        left: 10%;
        animation-delay: -5s;
    }

    .particles-bg::after {
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.06) 0%, transparent 70%);
        bottom: 10%;
        right: 10%;
        animation-delay: -10s;
    }

    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -40px) scale(1.05); }
        50% { transform: translate(-20px, 20px) scale(0.95); }
        75% { transform: translate(40px, 30px) scale(1.02); }
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-purple), var(--accent-pink));
        border-radius: 10px;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0820 0%, #150f2e 50%, #0d0820 100%) !important;
        border-right: 1px solid rgba(168, 85, 247, 0.15) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* ===== HERO SECTION ===== */
    .hero-container {
        text-align: center;
        padding: 1.5rem 1rem 2rem;
        position: relative;
        animation: fadeInDown 0.8s ease-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(236,72,153,0.1));
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 50px;
        padding: 6px 18px;
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--accent-purple);
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .hero-title {
        font-family: var(--font-display) !important;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6, #818cf8, #22d3ee);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 4s ease infinite;
        line-height: 1.15;
        margin-bottom: 0.75rem;
        letter-spacing: -1px;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    .hero-subtitle strong {
        color: var(--accent-purple);
        font-weight: 600;
    }

    /* ===== GLASS CARD ===== */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: slideUp 0.6s ease-out;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(168,85,247,0.4), rgba(236,72,153,0.3), transparent);
    }

    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: rgba(168, 85, 247, 0.35);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.25rem;
    }

    .card-icon {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }

    .card-icon-purple {
        background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(139,92,246,0.15));
        border: 1px solid rgba(168,85,247,0.3);
    }

    .card-icon-pink {
        background: linear-gradient(135deg, rgba(236,72,153,0.2), rgba(219,39,119,0.15));
        border: 1px solid rgba(236,72,153,0.3);
    }

    .card-icon-blue {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(79,70,229,0.15));
        border: 1px solid rgba(99,102,241,0.3);
    }

    .card-icon-emerald {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.15));
        border: 1px solid rgba(16,185,129,0.3);
    }

    .card-title {
        font-family: var(--font-display) !important;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }

    .card-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ===== PROCESS BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        font-family: var(--font-main) !important;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.3) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 35px rgba(168, 85, 247, 0.45) !important;
        background: linear-gradient(135deg, #9333ea, #db2777) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ===== TEXT AREA & FILE UPLOADER ===== */
    .stTextArea textarea {
        background: rgba(15, 10, 30, 0.6) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-main) !important;
        font-size: 0.92rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        resize: none !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.15) !important;
    }

    .stTextArea label {
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* File Uploader */
    section[data-testid="stFileUploader"] {
        background: rgba(15, 10, 30, 0.4);
        border: 2px dashed rgba(168, 85, 247, 0.25);
        border-radius: 16px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    section[data-testid="stFileUploader"]:hover {
        border-color: rgba(168, 85, 247, 0.5);
        background: rgba(20, 14, 40, 0.5);
    }

    section[data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* ===== PIPELINE PROGRESS ===== */
    .pipeline-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        font-family: var(--font-main);
        transition: all 0.4s ease;
    }

    .pipeline-step.inactive {
        background: rgba(30, 20, 50, 0.5);
        color: var(--text-muted);
        border: 1px solid rgba(100, 116, 139, 0.2);
    }

    .pipeline-step.active {
        background: linear-gradient(135deg, rgba(168,85,247,0.2), rgba(236,72,153,0.15));
        color: var(--accent-purple);
        border: 1px solid rgba(168,85,247,0.4);
        animation: pulseGlow 2s ease-in-out infinite;
    }

    .pipeline-step.done {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
        color: var(--accent-emerald);
        border: 1px solid rgba(16,185,129,0.3);
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 10px rgba(168,85,247,0.1); }
        50% { box-shadow: 0 0 25px rgba(168,85,247,0.25); }
    }

    .pipeline-arrow {
        color: var(--text-muted);
        font-size: 1.2rem;
        margin: 0 4px;
    }

    .step-icon {
        font-size: 1.1rem;
    }

    /* ===== SCORE GAUGE ===== */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem 1rem;
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }

    .gauge-ring {
        position: relative;
        width: 200px;
        height: 200px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .gauge-ring svg {
        position: absolute;
        top: 0;
        left: 0;
        width: 200px;
        height: 200px;
        transform: rotate(-90deg);
    }

    .gauge-ring svg circle {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
    }

    .gauge-bg {
        stroke: rgba(168, 85, 247, 0.1);
    }

    .gauge-fill {
        stroke: url(#gaugeGradient);
        stroke-dasharray: 565.48;
        animation: fillGauge 1.5s ease-out forwards;
    }

    @keyframes fillGauge {
        from { stroke-dashoffset: 565.48; }
    }

    .gauge-inner {
        text-align: center;
        z-index: 1;
    }

    .gauge-value {
        font-family: var(--font-display);
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }

    .gauge-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 4px;
        font-weight: 500;
    }

    .gauge-status {
        margin-top: 1rem;
        padding: 6px 20px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .gauge-status.excellent {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
        color: #34d399;
        border: 1px solid rgba(16,185,129,0.3);
    }

    .gauge-status.good {
        background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(139,92,246,0.1));
        color: #c084fc;
        border: 1px solid rgba(168,85,247,0.3);
    }

    .gauge-status.fair {
        background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(202,138,4,0.1));
        color: #fbbf24;
        border: 1px solid rgba(234,179,8,0.3);
    }

    .gauge-status.low {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.1));
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
    }

    /* ===== SKILL CHIPS ===== */
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.75rem;
    }

    .skill-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 16px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
        font-family: var(--font-main);
        animation: chipPop 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        transition: all 0.2s ease;
    }

    .skill-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.2);
    }

    .chip-purple {
        background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(139,92,246,0.1));
        color: #c084fc;
        border: 1px solid rgba(168,85,247,0.3);
    }

    .chip-pink {
        background: linear-gradient(135deg, rgba(236,72,153,0.15), rgba(219,39,119,0.1));
        color: #f472b6;
        border: 1px solid rgba(236,72,153,0.3);
    }

    .chip-blue {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(79,70,229,0.1));
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,0.3);
    }

    .chip-cyan {
        background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(6,182,212,0.1));
        color: #67e8f9;
        border: 1px solid rgba(34,211,238,0.3);
    }

    .chip-emerald {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.3);
    }

    @keyframes chipPop {
        from { opacity: 0; transform: scale(0.7); }
        to { opacity: 1; transform: scale(1); }
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(20, 14, 40, 0.5) !important;
        border: 1px solid rgba(168, 85, 247, 0.15) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    .streamlit-expanderContent {
        background: rgba(15, 10, 30, 0.4) !important;
        border: 1px solid rgba(168, 85, 247, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* ===== SUCCESS/ERROR MESSAGES ===== */
    .stSuccess, .stAlert {
        border-radius: 14px !important;
    }

    /* ===== TECH STACK BADGE ===== */
    .tech-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 3px 2px;
        background: rgba(20, 14, 40, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.2);
        color: var(--text-secondary);
    }

    /* ===== DIVIDER ===== */
    .gradient-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(168,85,247,0.3), rgba(236,72,153,0.2), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ===== RESULT SECTION ===== */
    .result-header {
        text-align: center;
        margin-bottom: 1.5rem;
        animation: fadeInDown 0.6s ease-out;
    }

    .result-title {
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .result-subtitle {
        font-size: 0.88rem;
        color: var(--text-muted);
    }

    /* ===== SIDEBAR SPECIFIC ===== */
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }

    .sidebar-logo-text {
        font-family: var(--font-display);
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sidebar-version {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: 2px;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.25rem 0 0.5rem;
    }

    .pipeline-info {
        padding: 12px 14px;
        background: rgba(15, 10, 30, 0.5);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .pipeline-info-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 3px;
    }

    .pipeline-info-desc {
        font-size: 0.73rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    .sidebar-footer {
        text-align: center;
        padding: 1rem 0;
        border-top: 1px solid rgba(168, 85, 247, 0.1);
        margin-top: 2rem;
    }

    .sidebar-footer p {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin: 2px 0;
    }

    /* ===== IMAGE PREVIEW ===== */
    .image-preview-container {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(168, 85, 247, 0.2);
        margin-top: 0.75rem;
    }

    .stImage {
        border-radius: 14px;
        overflow: hidden;
    }

    /* ===== METRICS ROW ===== */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 1rem;
    }

    .metric-card {
        background: rgba(15, 10, 30, 0.5);
        border: 1px solid rgba(168, 85, 247, 0.12);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-family: var(--font-display);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .metric-label {
        font-size: 0.73rem;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* ===== LOADING ANIMATION ===== */
    .stSpinner > div {
        border-color: var(--accent-purple) transparent transparent transparent !important;
    }

    /* ===== ANIMATION DELAYS ===== */
    .delay-1 { animation-delay: 0.1s !important; }
    .delay-2 { animation-delay: 0.2s !important; }
    .delay-3 { animation-delay: 0.3s !important; }
    .delay-4 { animation-delay: 0.4s !important; }
    .delay-5 { animation-delay: 0.5s !important; }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-subtitle { font-size: 0.9rem; }
        .pipeline-container { flex-direction: column; align-items: stretch; }
        .pipeline-arrow { transform: rotate(90deg); text-align: center; }
        .gauge-ring { width: 160px; height: 160px; }
        .gauge-ring svg { width: 160px; height: 160px; }
        .gauge-value { font-size: 2.2rem; }
        .metrics-row { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Particles background
st.markdown('<div class="particles-bg"></div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">🧠 RecruitAI</div>
        <div class="sidebar-version">v2.0 — Smart Talent Matching</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Pipeline info
    st.markdown('<div class="sidebar-section-title">📋 Pipeline AI</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pipeline-info">
        <div class="pipeline-info-title">1. 🔍 OCR Engine</div>
        <div class="pipeline-info-desc">Dual-engine: Tesseract v5 + PaddleOCR untuk ekstraksi teks maksimal dari gambar CV</div>
    </div>
    <div class="pipeline-info">
        <div class="pipeline-info-title">2. 🏷️ Dual NER Models</div>
        <div class="pipeline-info-desc">
            - <strong>WikiANN Model</strong>: Ekstraksi Nama, Institusi, dan Lokasi kandidat.<br>
            - <strong>Rekrutmen Model</strong>: Ekstraksi Keterampilan (Skills) dan Jabatan profesional.
        </div>
    </div>
    <div class="pipeline-info">
        <div class="pipeline-info-title">3. 🧬 Semantic Matching</div>
        <div class="pipeline-info-desc">Sentence Transformer (paraphrase-multilingual) menghitung cosine similarity antar embedding</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Tech Stack
    st.markdown('<div class="sidebar-section-title">⚡ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div>
        <span class="tech-badge">🤗 Transformers</span>
        <span class="tech-badge">🔤 WikiANN-BERT</span>
        <span class="tech-badge">🔤 IndoBERT-NER</span>
        <span class="tech-badge">📐 Sentence-BERT</span>
        <span class="tech-badge">👁️ PaddleOCR</span>
        <span class="tech-badge">🔍 Tesseract</span>
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">🎈 Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="sidebar-footer">
        <p>Dibuat dengan ❤️ untuk NLP</p>
        <p>Semester 6 — 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# LOAD MODELS (Cached)
# ==========================================
@st.cache_resource
def load_models():
    return load_ai_models()

try:
    with st.spinner("⏳ Memuat model AI..."):
        wikiann_pipe, rekrutmen_pipe, embedder = load_models()
except FileNotFoundError as e:
    st.error(f"🚨 {str(e)}")
    st.stop()

# ==========================================
# HERO SECTION
# ==========================================
# Lottie animation
lottie_ai = load_lottie_url("https://lottie.host/49952525-d686-4867-b4d0-c1e82e2528da/IMadXAp19G.json")

hero_col1, hero_col2, hero_col3 = st.columns([1, 3, 1])
with hero_col2:
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="hero_lottie")

    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">✨ AI-Powered Recruitment System</div>
        <h1 class="hero-title">Smart Talent Matching</h1>
        <p class="hero-subtitle">
            Ekstraksi profil dan keterampilan menggunakan <strong>OCR</strong> & <strong>Dual NER Models</strong>, 
            lalu cocokkan dengan kriteria lowongan menggunakan <strong>Semantic AI</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ==========================================
# PIPELINE VISUALIZATION
# ==========================================
st.markdown("""
<div class="pipeline-container">
    <div class="pipeline-step inactive">
        <span class="step-icon">📄</span> Upload CV
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step inactive">
        <span class="step-icon">🔍</span> OCR
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step inactive">
        <span class="step-icon">🏷️</span> NER
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step inactive">
        <span class="step-icon">🧬</span> Matching
    </div>
    <span class="pipeline-arrow">→</span>
    <div class="pipeline-step inactive">
        <span class="step-icon">📊</span> Hasil
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon card-icon-purple">📝</div>
            <div>
                <p class="card-title">Kriteria Lowongan</p>
                <p class="card-subtitle">Pilih posisi atau tulis deskripsi sendiri</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dictionary posisi pekerjaan beserta deskripsi kriterianya
    predefined_jobs = {
        "✏️ Kustom (Tulis Sendiri)": "",
        "💻 Data Scientist": "Mencari Data Scientist yang menguasai Python, SQL, Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch, Pandas, NumPy, dan analisis data statistik.",
        "🌐 Frontend Developer": "Mencari Frontend Developer yang ahli menggunakan HTML, CSS, JavaScript, ReactJS, Next.js, TypeScript, Tailwind CSS, dan memahami prinsip UI/UX.",
        "🔧 Backend Developer": "Mencari Backend Developer yang menguasai Node.js, Python, Golang, PostgreSQL, MySQL, REST API, Docker, dan arsitektur Microservices.",
        "📱 Mobile Developer": "Mencari Mobile Developer yang berpengalaman dengan Flutter, Dart, React Native, Firebase, REST API, dan pengembangan aplikasi Android maupun iOS.",
        "🎨 UI/UX Designer": "Mencari desainer yang mahir menggunakan Figma, Adobe XD, membuat wireframe, prototyping interaktif, dan riset pengguna (user research).",
        "☁️ DevOps Engineer": "Mencari DevOps Engineer yang menguasai Docker, Kubernetes, AWS, GCP, Azure, CI/CD pipeline, Linux, dan otomatisasi infrastruktur.",
        "🤖 Machine Learning Engineer": "Mencari Machine Learning Engineer yang berpengalaman membangun model ML, menguasai Python, Scikit-learn, TensorFlow, PyTorch, serta deployment model ke produksi.",
        "🗄️ Database Administrator": "Mencari DBA yang menguasai MySQL, PostgreSQL, MongoDB, Redis, query optimization, database backup, dan manajemen keamanan basis data.",
    }

    selected_job = st.selectbox(
        "Pilih Posisi Pekerjaan:",
        options=list(predefined_jobs.keys()),
        label_visibility="collapsed"
    )

    # Auto-fill deskripsi berdasarkan pilihan
    default_text = predefined_jobs[selected_job] if predefined_jobs[selected_job] else "Tuliskan kriteria keterampilan dan pengalaman yang Anda cari..."

    job_desc = st.text_area(
        "Masukkan kriteria yang dicari:",
        value=default_text,
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon card-icon-pink">📄</div>
            <div>
                <p class="card-title">Upload CV Pelamar</p>
                <p class="card-subtitle">Upload hasil scan CV dalam format JPG/PNG</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload CV (JPG/PNG/PDF)", 
        type=["jpg", "jpeg", "png", "pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".pdf"):
            st.info(f"📄 PDF Uploaded: {uploaded_file.name}")
        else:
            st.image(uploaded_file, caption="Preview CV", use_column_width=True)

# ==========================================
# PROCESS BUTTON
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    process_btn = st.button("🚀 Proses CV & Hitung Kecocokan", disabled=not uploaded_file)

# ==========================================
# PROCESSING PIPELINE
# ==========================================
if uploaded_file and process_btn:
    # Save temp image
    temp_image_path = "temp_cv.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Progress pipeline - OCR active
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step done"><span class="step-icon">✅</span> Upload CV</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active"><span class="step-icon">🔍</span> OCR</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">🏷️</span> NER</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">🧬</span> Matching</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">📊</span> Hasil</div>
    </div>
    """, unsafe_allow_html=True)

    # PHASE 1: OCR / Text Extraction
    with st.spinner("🔍 Mengekstrak teks dari dokumen..."):
        if uploaded_file.name.lower().endswith('.pdf'):
            raw_text, extraction_method = process_pdf(temp_image_path)
            st.info(f"💡 Metode Ekstraksi PDF: {extraction_method}")
        else:
            raw_text = extract_text_from_image(temp_image_path)
    # Progress pipeline - NER active
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step done"><span class="step-icon">✅</span> Upload CV</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step done"><span class="step-icon">✅</span> OCR</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active"><span class="step-icon">🏷️</span> NER</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">🧬</span> Matching</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">📊</span> Hasil</div>
    </div>
    """, unsafe_allow_html=True)

    # PHASE 2: NER
    with st.spinner("🏷️ Menganalisis entitas penting (Dual NER)..."):
        names, institutions, locations, extracted_skills, extracted_jobs, entities = run_ner_extraction(raw_text, wikiann_pipe, rekrutmen_pipe)
    # Progress pipeline - Matching active
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step done"><span class="step-icon">✅</span> Upload CV</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step done"><span class="step-icon">✅</span> OCR</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step done"><span class="step-icon">✅</span> NER</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step active"><span class="step-icon">🧬</span> Matching</div>
        <span class="pipeline-arrow">→</span>
        <div class="pipeline-step inactive"><span class="step-icon">📊</span> Hasil</div>
    </div>
    """, unsafe_allow_html=True)

    # PHASE 3: Embedding & Matching
    with st.spinner("🧬 Menghitung skor kecocokan (Embedding)..."):
        if not raw_text.strip():
            st.error("⚠️ Teks dokumen kosong atau tidak terbaca. Silakan unggah dokumen yang lebih jelas.")
        else:
            match_percentage, cosine_score, keyword_score = calculate_similarity(extracted_jobs, extracted_skills, job_desc, raw_text, embedder)
            contact_info = extract_contact_info(raw_text)
            # Final pipeline - all done
            st.markdown("""
            <div class="pipeline-container">
                <div class="pipeline-step done"><span class="step-icon">✅</span> Upload CV</div>
                <span class="pipeline-arrow">→</span>
                <div class="pipeline-step done"><span class="step-icon">✅</span> OCR</div>
                <span class="pipeline-arrow">→</span>
                <div class="pipeline-step done"><span class="step-icon">✅</span> NER</div>
                <span class="pipeline-arrow">→</span>
                <div class="pipeline-step done"><span class="step-icon">✅</span> Matching</div>
                <span class="pipeline-arrow">→</span>
                <div class="pipeline-step done"><span class="step-icon">✅</span> Hasil</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            # ==========================================
            # RESULTS SECTION
            # ==========================================
            st.markdown("""
            <div class="result-header">
                <div class="result-title">📊 Hasil Analisis</div>
                <div class="result-subtitle">Pemrosesan selesai — berikut hasilnya</div>
            </div>
            """, unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1, 1], gap="large")

            with res_col1:
                # Determine score status
                if match_percentage >= 75:
                    status_class = "excellent"
                    status_text = "Sangat Cocok ✨"
                    gauge_color_start = "#10b981"
                    gauge_color_end = "#34d399"
                elif match_percentage >= 55:
                    status_class = "good"
                    status_text = "Cukup Cocok 👍"
                    gauge_color_start = "#a855f7"
                    gauge_color_end = "#c084fc"
                elif match_percentage >= 35:
                    status_class = "fair"
                    status_text = "Kurang Cocok ⚡"
                    gauge_color_start = "#eab308"
                    gauge_color_end = "#fbbf24"
                else:
                    status_class = "low"
                    status_text = "Tidak Cocok ❌"
                    gauge_color_start = "#ef4444"
                    gauge_color_end = "#f87171"

                # Circumference = 2 * π * r = 2 * π * 90 ≈ 565.48
                circumference = 565.48
                dash_offset = circumference - (circumference * match_percentage / 100)

                st.markdown(f"""
                <div class="glass-card" style="animation-delay: 0.1s">
                    <div class="gauge-container">
                        <div class="gauge-ring">
                            <svg viewBox="0 0 200 200">
                                <defs>
                                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" style="stop-color:{gauge_color_start}" />
                                        <stop offset="100%" style="stop-color:{gauge_color_end}" />
                                    </linearGradient>
                                </defs>
                                <circle class="gauge-bg" cx="100" cy="100" r="90" />
                                <circle class="gauge-fill" cx="100" cy="100" r="90" 
                                    style="stroke-dashoffset: {dash_offset};" />
                            </svg>
                            <div class="gauge-inner">
                                <div class="gauge-value">{match_percentage:.1f}%</div>
                                <div class="gauge-label">Kecocokan (Semantic)</div>
                            </div>
                        </div>
                        <div class="gauge-status {status_class}">{status_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                # Biographical Card (WikiANN + Regex)
                candidate_name = ", ".join(names) if names else "Tidak Terdeteksi"
                candidate_inst = ", ".join(institutions) if institutions else "Tidak Terdeteksi"
                candidate_loc = ", ".join(locations) if locations else "Tidak Terdeteksi"
                candidate_email = contact_info.get('email') or "Tidak Terdeteksi"
                candidate_phone = contact_info.get('phone') or "Tidak Terdeteksi"
                
                st.markdown(f"""
                <div class="glass-card" style="animation-delay: 0.15s">
                    <div class="card-header">
                        <div class="card-icon card-icon-blue">👤</div>
                        <div>
                            <p class="card-title">Profil Pelamar</p>
                            <p class="card-subtitle">Entitas profil diekstrak oleh WikiANN Model + Regex</p>
                        </div>
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.9; color: var(--text-primary);">
                        <strong>👤 Nama:</strong> {candidate_name}<br>
                        <strong>🏢 Universitas/Institusi:</strong> {candidate_inst}<br>
                        <strong>📍 Lokasi:</strong> {candidate_loc}<br>
                        <strong>📧 Email:</strong> {candidate_email}<br>
                        <strong>📞 Telepon:</strong> {candidate_phone}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Skills & Jobs Card (Rekrutmen Model)
                st.markdown(f"""
                <div class="glass-card" style="animation-delay: 0.25s">
                    <div class="card-header">
                        <div class="card-icon card-icon-emerald">🏷️</div>
                        <div>
                            <p class="card-title">Keterampilan & Pengalaman Kerja</p>
                            <p class="card-subtitle">{len(extracted_skills)} keterampilan & {len(extracted_jobs)} jabatan terdeteksi</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if extracted_jobs:
                    st.markdown("""
                    <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; margin-top: 10px;">
                        💼 POSISI / JABATAN TERDETEKSI:
                    </div>
                    """, unsafe_allow_html=True)
                    job_chips = ""
                    for i, job in enumerate(extracted_jobs):
                        delay = f"animation-delay: {0.05 * (i+1)}s"
                        job_chips += f'<span class="skill-chip chip-pink" style="{delay}">💼 {job}</span>'
                    st.markdown(f'<div class="skills-container">{job_chips}</div>', unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; margin-top: 15px;">
                    ⚡ KETERAMPILAN TERDETEKSI:
                </div>
                """, unsafe_allow_html=True)
                
                chip_colors = ["chip-purple", "chip-blue", "chip-cyan", "chip-emerald"]
                chips_html = ""
                for i, skill in enumerate(extracted_skills):
                    color = chip_colors[i % len(chip_colors)]
                    delay = f"animation-delay: {0.05 * (i+1)}s"
                    chips_html += f'<span class="skill-chip {color}" style="{delay}">⚡ {skill}</span>'

                if not extracted_skills:
                    chips_html = '<span style="color: var(--text-muted); font-size: 0.85rem; font-style: italic;">Tidak ada keterampilan terdeteksi</span>'

                st.markdown(f"""
                    <div class="skills-container">
                        {chips_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row — Score breakdown + stats
            word_count = len(raw_text.split())
            entity_count = len(entities)
            skill_count = len(extracted_skills)

            st.markdown(f"""
            <div class="metrics-row" style="grid-template-columns: repeat(4, 1fr);">
                <div class="metric-card">
                    <div class="metric-value" style="color: #818cf8">{match_percentage:.1f}%</div>
                    <div class="metric-label">Semantic Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #f472b6">{entity_count}</div>
                    <div class="metric-label">Entitas NER</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #34d399">{skill_count}</div>
                    <div class="metric-label">Keterampilan</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="color: #fbbf24">{word_count}</div>
                    <div class="metric-label">Kata (CV)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # OCR raw text expander
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📄 Lihat Teks Mentah Hasil OCR"):
                st.code(raw_text, language=None)