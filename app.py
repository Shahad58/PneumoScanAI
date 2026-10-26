import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import os
import io
import base64
import time
import random
from datetime import datetime, timezone
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────
DATA_DIR = "chest_xray"
MODEL_PATH = "pneumonia_mobilenet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Streamlit renamed use_container_width -> width="stretch" (1.50+). Support both.
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (0, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# ──────────────────────────────────────────────────────────────
#  Session state
# ──────────────────────────────────────────────────────────────
defaults = {
    "lang": "en",
    "analysis_result": None,
    "uploaded_image": None,
    "uploader_key": 0,
    "last_sig": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
#  Texts
# ──────────────────────────────────────────────────────────────
TEXTS = {
    "en": {
        "lang_btn": "🌐 العربية",
        "tag": "Clinical AI Diagnostics",
        "subtitle": "Intelligent Pneumonia Detection",
        "desc": "Upload a chest X-ray or try a demo case to receive an AI-powered diagnosis with an explainable heatmap.",
        "feat1": "Explainable heatmap",
        "feat2": "Results in seconds",
        "feat3": "PDF clinical report",
        "form_title": "Patient Intake Form",
        "doctor_name": "Attending Doctor",
        "patient_name": "Patient Name",
        "patient_id": "Patient ID",
        "upload_label": "Upload X-Ray Image",
        "or": "or",
        "demo_btn": "🎲 Try a Demo Case",
        "analyze_btn": "🔬 Run AI Analysis",
        "reanalyze_btn": "🔄 New Analysis",
        "original": "Original X-Ray",
        "heatmap": "AI Focus Area (Grad-CAM)",
        "legend_low": "Low focus",
        "legend_high": "High focus",
        "prediction": "Final Diagnosis",
        "confidence": "Confidence Score",
        "normal": "Normal",
        "pneumonia": "Pneumonia Detected",
        "howto_title": "Clinical Note",
        "howto_desc": "The heatmap highlights the regions the AI analyzed. Red zones indicate high suspicion of opacity, a key indicator of pneumonia.",
        "disclaimer": "System is for educational demonstration. Always consult a licensed radiologist.",
        "analyzing": "Scanning the X-ray…",
        "missing_model": "Demo Mode: Model file not found. Showing simulated result for illustration.",
        "missing_data": "Dataset not found. Place the chest_xray folder next to app.py to enable demo cases.",
        "download_report": "📄 Download Clinical PDF Report",
        "status_ready": "Waiting for an X-ray",
        "status_ready_sub": "Upload an image or load a demo case, then run the analysis.",
        "status_demo": "Demo case loaded. Ready to analyze.",
        "status_uploaded": "Image uploaded. Ready to analyze.",
        "made_with": "Created with",
        "by": "by",
    },
    "ar": {
        "lang_btn": "🌐 English",
        "tag": "تشخيص طبي بالذكاء الاصطناعي",
        "subtitle": "التشخيص الذكي للالتهاب الرئوي",
        "desc": "ارفع صورة أشعة سينية أو جرّب حالة تجريبية للحصول على تشخيص بالذكاء الاصطناعي مع خريطة حرارية توضيحية.",
        "feat1": "خريطة حرارية توضيحية",
        "feat2": "نتيجة خلال ثوانٍ",
        "feat3": "تقرير طبي PDF",
        "form_title": "نموذج بيانات المريض",
        "doctor_name": "الطبيب المعالج",
        "patient_name": "اسم المريض",
        "patient_id": "رقم المريض",
        "upload_label": "رفع صورة الأشعة السينية",
        "or": "أو",
        "demo_btn": "🎲 تجربة حالة سريعة",
        "analyze_btn": "🔬 تشغيل التحليل",
        "reanalyze_btn": "🔄 تحليل جديد",
        "original": "الصورة الأصلية",
        "heatmap": "منطقة تركيز الذكاء الاصطناعي",
        "legend_low": "تركيز منخفض",
        "legend_high": "تركيز عالٍ",
        "prediction": "التشخيص النهائي",
        "confidence": "نسبة الثقة",
        "normal": "طبيعي",
        "pneumonia": "التهاب رئوي",
        "howto_title": "ملاحظة طبية",
        "howto_desc": "الخريطة الحرارية توضح المناطق التي حللها الذكاء الاصطناعي. المناطق الحمراء تشير إلى اشتباه عالٍ، وهو مؤشر رئيسي للالتهاب الرئوي.",
        "disclaimer": "النظام مخصص للعرض التعليمي. يجب دائماً استشارة طبيب أشعة مرخص.",
        "analyzing": "جاري فحص الأشعة…",
        "missing_model": "وضع المحاكاة: لم يتم العثور على ملف النموذج. يتم عرض نتيجة توضيحية.",
        "missing_data": "لم يتم العثور على البيانات. ضع مجلد chest_xray بجانب app.py لتفعيل الحالات التجريبية.",
        "download_report": "📄 تحميل التقرير الطبي (PDF)",
        "status_ready": "بانتظار صورة الأشعة",
        "status_ready_sub": "ارفع صورة أو حمّل حالة تجريبية ثم شغّل التحليل.",
        "status_demo": "تم تحميل الحالة التجريبية. جاهز للتحليل.",
        "status_uploaded": "تم رفع الصورة. جاهز للتحليل.",
        "made_with": "صُنع بـ",
        "by": "بواسطة",
    },
}

# The PDF uses built-in Helvetica which has no Arabic glyphs, so reports are always in English.
PDF_T = {
    "report_title": "PneumoScan AI - Clinical Report",
    "field_date": "Date",
    "field_doctor": "Doctor",
    "field_patient": "Patient Name",
    "field_patient_id": "Patient ID",
    "field_prediction": "Diagnosis",
    "field_confidence": "Confidence",
    "field_original": "Original X-Ray",
    "field_heatmap": "Grad-CAM Heatmap",
    "form_title": "Patient Information",
    "prediction": "Final Diagnosis",
    "heatmap": "AI Focus Area (Grad-CAM)",
    "normal": "Normal",
    "pneumonia": "Pneumonia Detected",
    "disclaimer_title": "Disclaimer",
    "report_disclaimer_text": "This AI-assisted analysis does not replace professional medical diagnosis.",
}

t = TEXTS[st.session_state.lang]
is_rtl = st.session_state.lang == "ar"
direction = "rtl" if is_rtl else "ltr"
font_display = "Cairo" if is_rtl else "Sora"
font_body = "Cairo" if is_rtl else "Plus Jakarta Sans"

# ──────────────────────────────────────────────────────────────
#  Styles
# ──────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Cairo:wght@400;600;700;800&display=swap');

:root {
    --navy:#0B2A4A; --ink:#16324F; --muted:#5B7083; --line:#D5E5EA;
    --teal:#0F8B8D; --aqua:#3CCFCF; --mint:#2BB673; --coral:#E5484D;
    --f-display:'__FD__', sans-serif; --f-body:'__FB__', sans-serif;
}

.stApp {
    background-color:#EAF4F6;
    background-image:
        radial-gradient(circle at 12% 18%, rgba(60,207,207,.16), transparent 34%),
        radial-gradient(circle at 88% 12%, rgba(15,139,141,.10), transparent 30%),
        radial-gradient(circle at 70% 90%, rgba(11,42,74,.07), transparent 36%);
    font-family:var(--f-body);
    color:var(--ink);
}
#MainMenu, footer, header {visibility:hidden;}
html, body, .stApp, .stMarkdown, p, label, button, input, textarea, li, h1, h2, h3, h4 { font-family:var(--f-body); }
.block-container { padding-top:1.4rem; max-width:1240px; position:relative; z-index:1; }

/* ═════════ Living background ═════════ */
.bg-blob { position:fixed; border-radius:50%; pointer-events:none; z-index:-1; filter:blur(10px); }
.bg-blob.b1 { width:560px; height:560px; top:-160px; left:-140px;
    background:radial-gradient(circle, rgba(60,207,207,.30), transparent 66%);
    animation:drift1 24s ease-in-out infinite alternate; }
.bg-blob.b2 { width:620px; height:620px; bottom:-220px; right:-160px;
    background:radial-gradient(circle, rgba(15,139,141,.22), transparent 66%);
    animation:drift2 28s ease-in-out infinite alternate; }
@keyframes drift1 { to { transform:translate(140px, 90px) scale(1.15); } }
@keyframes drift2 { to { transform:translate(-120px, -80px) scale(1.12); } }

.bubble { position:fixed; bottom:-90px; border-radius:50%; pointer-events:none; z-index:-1;
    background:radial-gradient(circle at 30% 28%, rgba(255,255,255,.95), rgba(60,207,207,.22) 46%, rgba(15,139,141,.10));
    border:1px solid rgba(255,255,255,.75);
    box-shadow:inset -4px -6px 12px rgba(15,139,141,.10);
    opacity:0; animation:rise linear infinite; }
@keyframes rise {
    0%   { transform:translate(0,0) scale(.8); opacity:0; }
    12%  { opacity:.9; }
    50%  { transform:translate(26px,-55vh) scale(1); }
    100% { transform:translate(-14px,-118vh) scale(1.1); opacity:0; }
}

/* ═════════ Hero ═════════ */
.hero { text-align:center; margin:.4rem auto 2.4rem; position:relative; }

.logo-badge { width:92px; height:92px; margin:0 auto 14px; animation:floaty 5s ease-in-out infinite;
    filter:drop-shadow(0 14px 22px rgba(11,42,74,.28)); }
.logo-badge svg { width:100%; height:100%; display:block; overflow:visible; }
@keyframes floaty { 50% { transform:translateY(-7px); } }
.lung { transform-box:fill-box; transform-origin:50% 60%; animation:breathe 4.2s ease-in-out infinite; }
.lung.r { animation-delay:.06s; }
@keyframes breathe { 50% { transform:scale(1.07); } }
.ecg-mini { stroke-dasharray:140; stroke-dashoffset:140; animation:ecgmini 2.6s ease-in-out infinite; }
@keyframes ecgmini { 0%{stroke-dashoffset:140;} 55%{stroke-dashoffset:0;} 100%{stroke-dashoffset:-140;} }

.hero-tag { display:inline-flex; align-items:center; gap:9px; font-size:13px; font-weight:700; color:var(--teal);
    background:rgba(255,255,255,.75); border:1px solid rgba(15,139,141,.22); padding:7px 18px; border-radius:30px;
    margin-bottom:14px; backdrop-filter:blur(8px); }
.live-dot { width:8px; height:8px; border-radius:50%; background:var(--mint); position:relative; }
.live-dot::after { content:''; position:absolute; inset:-5px; border-radius:50%; border:2px solid var(--mint);
    animation:ping 1.8s ease-out infinite; }
@keyframes ping { from { transform:scale(.5); opacity:.9; } to { transform:scale(1.7); opacity:0; } }

.hero-title { font-family:var(--f-display); font-weight:800; font-size:56px; letter-spacing:-1.5px; color:var(--navy);
    line-height:1.05; margin-bottom:6px; direction:ltr; }
.hero-title .scan { color:var(--teal); }
.hero-title .ai { display:inline-block; font-size:.34em; letter-spacing:.5px; vertical-align:top; margin-inline-start:8px;
    margin-top:.45em; padding:4px 11px; border-radius:10px; color:#fff; font-weight:700;
    background:linear-gradient(135deg,var(--teal),var(--navy)); box-shadow:0 6px 16px rgba(15,139,141,.35);
    animation:badgeglow 3s ease-in-out infinite; }
@keyframes badgeglow { 50% { box-shadow:0 6px 26px rgba(60,207,207,.75); } }

.hero-sub { font-family:var(--f-display); font-size:20px; font-weight:600; color:var(--ink); margin-bottom:8px; }

.ecg { display:block; width:min(520px,86%); height:56px; margin:6px auto 4px; overflow:visible; }
.ecg .base { stroke:rgba(15,139,141,.16); }
.ecg .trace { stroke-dasharray:1000; stroke-dashoffset:1000; animation:ecgrun 3.4s ease-in-out infinite;
    filter:drop-shadow(0 0 5px rgba(60,207,207,.9)); }
@keyframes ecgrun { 0%{stroke-dashoffset:1000;} 55%{stroke-dashoffset:0;} 100%{stroke-dashoffset:-1000;} }

.hero-desc { max-width:620px; margin:6px auto 18px; color:var(--muted); font-size:16px; line-height:1.75; }

.feat-row { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; }
.feat { display:inline-flex; align-items:center; gap:8px; padding:8px 16px; font-size:13.5px; font-weight:600;
    color:var(--ink); background:rgba(255,255,255,.7); border:1px solid var(--line); border-radius:30px;
    backdrop-filter:blur(8px); transition:border-color .25s, background .25s; }
.feat:hover { border-color:var(--aqua); background:#fff; }
.feat svg { width:18px; height:18px; stroke:var(--teal); fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }

/* ═════════ Cards ═════════ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:rgba(255,255,255,.72) !important;
    backdrop-filter:blur(18px) !important; -webkit-backdrop-filter:blur(18px) !important;
    border-radius:26px !important; border:1px solid rgba(255,255,255,.9) !important;
    box-shadow:0 18px 50px -18px rgba(11,42,74,.20) !important;
    padding:1.8rem !important; transition:border-color .3s, box-shadow .3s;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color:rgba(60,207,207,.55) !important;
    box-shadow:0 22px 56px -16px rgba(15,139,141,.28) !important;
}

.form-title { display:flex; align-items:center; gap:12px; font-family:var(--f-display); font-size:20px; font-weight:700;
    color:var(--navy); margin-bottom:22px; padding-bottom:14px; border-bottom:2px solid rgba(15,139,141,.12); }
.form-title .ico { width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
    background:linear-gradient(135deg,var(--teal),var(--navy)); box-shadow:0 8px 18px rgba(15,139,141,.35); }
.form-title .ico svg { width:20px; height:20px; stroke:#fff; fill:none; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }

.divider-or { display:flex; align-items:center; gap:14px; color:#8CA2B0; font-weight:600; font-size:13px; margin:16px 0; }
.divider-or::before, .divider-or::after { content:''; flex:1; height:1px; background:var(--line); }

/* ═════════ Inputs ═════════ */
.stTextInput > div > div > input {
    background:#fff !important; border:2px solid var(--line) !important; border-radius:14px !important;
    color:var(--navy) !important; padding:13px 17px !important; font-size:15px !important; font-weight:500 !important;
    transition:border-color .25s, box-shadow .25s !important; direction:__DIR__; text-align:start;
}
.stTextInput > div > div > input:focus { border-color:var(--teal) !important; box-shadow:0 0 0 4px rgba(60,207,207,.22) !important; }
.stTextInput label, .stFileUploader label { color:var(--ink) !important; font-weight:700 !important; font-size:14px !important; }

[data-testid="stFileUploader"] section { background:rgba(255,255,255,.75); border:2px dashed #A9CDD4; border-radius:18px;
    transition:border-color .3s, background .3s; }
[data-testid="stFileUploader"] section:hover { border-color:var(--teal); background:rgba(60,207,207,.07); }

/* ═════════ Buttons ═════════ */
.stButton > button, .stDownloadButton > button {
    position:relative; overflow:hidden; color:#fff; border:none; border-radius:14px; padding:13px 26px; width:100%;
    font-family:var(--f-body); font-weight:700; font-size:15px;
    background:linear-gradient(135deg,var(--teal) 0%,var(--navy) 100%);
    box-shadow:0 10px 24px -8px rgba(11,42,74,.55);
    transition:transform .25s cubic-bezier(.2,.9,.3,1.3), box-shadow .25s;
}
.stButton > button::after, .stDownloadButton > button::after {
    content:''; position:absolute; top:0; left:-70%; width:50%; height:100%;
    background:linear-gradient(100deg, transparent, rgba(255,255,255,.38), transparent); transform:skewX(-20deg);
}
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-2px); color:#fff; border:none;
    box-shadow:0 14px 30px -8px rgba(15,139,141,.65); }
.stButton > button:hover::after, .stDownloadButton > button:hover::after { left:130%; transition:left .65s ease; }
.stButton > button:active { transform:translateY(0) scale(.99); }
.stButton > button p, .stDownloadButton > button p { font-weight:700; font-size:15px; color:#fff; }
.stDownloadButton > button { background:linear-gradient(135deg,var(--mint) 0%,#0B7A5A 100%); box-shadow:0 10px 24px -8px rgba(43,182,115,.6); }

/* ═════════ X-ray scan frame ═════════ */
.scan-wrap { margin:6px 0 4px; }
.scan-frame { position:relative; overflow:hidden; border-radius:18px; background:#04121F;
    box-shadow:0 14px 34px -10px rgba(4,18,31,.55); }
.scan-frame img { display:block; width:100%; height:auto; }
.scan-line { position:absolute; left:0; right:0; top:0; height:2px; pointer-events:none;
    background:linear-gradient(90deg, transparent, var(--aqua), transparent);
    box-shadow:0 0 18px 5px rgba(60,207,207,.55); animation:scan 3.6s ease-in-out infinite; }
.scan-frame.slow .scan-line { animation-duration:5.2s; }
@keyframes scan { 0%{top:0;opacity:0;} 8%{opacity:1;} 50%{top:calc(100% - 2px);opacity:1;} 92%{opacity:1;} 100%{top:0;opacity:0;} }
.corner { position:absolute; width:22px; height:22px; border:2px solid var(--aqua); pointer-events:none; opacity:.9; }
.corner.tl { top:10px; left:10px; border-right:0; border-bottom:0; border-top-left-radius:6px; }
.corner.tr { top:10px; right:10px; border-left:0; border-bottom:0; border-top-right-radius:6px; }
.corner.bl { bottom:10px; left:10px; border-right:0; border-top:0; border-bottom-left-radius:6px; }
.corner.br { bottom:10px; right:10px; border-left:0; border-top:0; border-bottom-right-radius:6px; }
.img-caption { text-align:center; font-size:14px; font-weight:700; color:var(--muted); margin:12px 0 4px; }

.legend { display:flex; align-items:center; gap:10px; font-size:12px; color:var(--muted); font-weight:600; margin-top:10px; }
.legend .bar { flex:1; height:8px; border-radius:8px;
    background:linear-gradient(90deg,#2B3FD6,#2BB6E6,#4FD17A,#F5D547,#F08A2E,#D9262C); }

/* ═════════ Loader ═════════ */
.loader { display:flex; align-items:center; gap:16px; padding:16px 20px; margin-top:14px; border-radius:18px;
    background:linear-gradient(135deg,rgba(15,139,141,.10),rgba(11,42,74,.07)); border:1px solid rgba(15,139,141,.22); }
.loader-ring { width:38px; height:38px; border-radius:50%; flex:none; border:4px solid rgba(15,139,141,.18);
    border-top-color:var(--teal); animation:spin .9s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.loader .txt { font-weight:700; color:var(--navy); font-size:15px; }
.loader .dots span { display:inline-block; width:5px; height:5px; margin:0 2px; border-radius:50%; background:var(--teal);
    animation:bounce 1s ease-in-out infinite; }
.loader .dots span:nth-child(2) { animation-delay:.15s; }
.loader .dots span:nth-child(3) { animation-delay:.3s; }
@keyframes bounce { 50% { transform:translateY(-6px); opacity:.4; } }

/* ═════════ Empty state ═════════ */
.empty { text-align:center; padding:56px 20px 46px; }
.empty .orb { position:relative; width:170px; height:170px; margin:0 auto 22px; }
.empty .orb .ripple { position:absolute; inset:0; border-radius:50%; border:2px solid rgba(60,207,207,.55);
    animation:ripple 3.6s ease-out infinite; }
.empty .orb .ripple:nth-child(2) { animation-delay:1.2s; }
.empty .orb .ripple:nth-child(3) { animation-delay:2.4s; }
@keyframes ripple { from { transform:scale(.45); opacity:.9; } to { transform:scale(1.25); opacity:0; } }
.empty .orb svg { position:absolute; inset:28px; width:calc(100% - 56px); height:calc(100% - 56px); overflow:visible;
    animation:floaty 5s ease-in-out infinite; }
.empty .lung-outline { fill:rgba(60,207,207,.12); stroke:var(--teal); stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round;
    stroke-dasharray:240; animation:draw 4.5s ease-in-out infinite; }
@keyframes draw { 0%{stroke-dashoffset:240;} 55%,100%{stroke-dashoffset:0;} }
.empty h3 { font-family:var(--f-display); color:var(--navy); font-weight:700; font-size:21px; margin-bottom:6px; }
.empty p { color:var(--muted); font-size:15px; max-width:340px; margin:0 auto; line-height:1.7; }

/* ═════════ Result ═════════ */
@property --num { syntax:'<integer>'; initial-value:0; inherits:false; }

.result-card { display:flex; align-items:center; gap:26px; padding:26px 28px; border-radius:22px; margin-bottom:22px;
    border:1px solid; position:relative; overflow:hidden; animation:pop .6s cubic-bezier(.2,.9,.3,1.25) both; }
@keyframes pop { from { transform:scale(.94); opacity:0; } to { transform:scale(1); opacity:1; } }
.res-normal { background:linear-gradient(135deg,#F0FDF6,#D8F6E5); border-color:#B5EBCB; }
.res-pneumonia { background:linear-gradient(135deg,#FFF3F3,#FFE0E1); border-color:#FFC4C6; animation:pop .6s cubic-bezier(.2,.9,.3,1.25) both, alert 2.4s ease-in-out .7s infinite; }
@keyframes alert { 0%,100% { box-shadow:0 0 0 0 rgba(229,72,77,0); } 50% { box-shadow:0 0 0 10px rgba(229,72,77,.13); } }

.ring-wrap { position:relative; width:128px; height:128px; flex:none; }
.ring { width:100%; height:100%; transform:rotate(-90deg); }
.ring-bg { fill:none; stroke:rgba(11,42,74,.09); stroke-width:10; }
.ring-fg { fill:none; stroke-width:10; stroke-linecap:round; stroke-dasharray:326.7; stroke-dashoffset:326.7;
    animation:fillring 1.5s cubic-bezier(.3,.8,.3,1) .2s forwards; }
.res-normal .ring-fg { stroke:var(--mint); }
.res-pneumonia .ring-fg { stroke:var(--coral); }
@keyframes fillring { to { stroke-dashoffset:var(--off); } }
.ring-num { position:absolute; inset:0; display:flex; align-items:baseline; justify-content:center; padding-top:41px;
    font-family:var(--f-display); font-weight:800; color:var(--navy); direction:ltr; }
.count { font-size:34px; counter-reset:num var(--num); animation:countup 1.5s cubic-bezier(.3,.8,.3,1) .2s forwards; }
.count::after { content:counter(num); }
@keyframes countup { from { --num:0; } to { --num:var(--target); } }
.ring-num small { font-size:15px; font-weight:700; opacity:.7; }

.res-text { flex:1; min-width:0; }
.res-label { font-size:13px; font-weight:700; color:var(--muted); margin-bottom:6px; }
.res-value { font-family:var(--f-display); font-size:32px; font-weight:800; letter-spacing:-.8px; line-height:1.15; margin-bottom:10px; }
.val-normal { color:#15803D; } .val-pneumonia { color:#C8262C; }
.res-conf { display:inline-flex; align-items:center; gap:8px; font-size:14px; font-weight:700; color:var(--ink);
    background:rgba(255,255,255,.7); padding:6px 14px; border-radius:20px; }
.res-conf svg { width:16px; height:16px; fill:none; stroke-width:3; stroke-linecap:round; stroke-linejoin:round; }
.res-normal .res-conf svg { stroke:var(--mint); stroke-dasharray:24; stroke-dashoffset:24; animation:tick .6s ease .9s forwards; }
.res-pneumonia .res-conf svg { stroke:var(--coral); animation:wiggle 1.6s ease-in-out infinite; }
@keyframes tick { to { stroke-dashoffset:0; } }
@keyframes wiggle { 0%,100% { transform:rotate(0); } 10%,30% { transform:rotate(-9deg); } 20%,40% { transform:rotate(9deg); } 50% { transform:rotate(0); } }

.info-box { background:linear-gradient(to right, rgba(15,139,141,.07), rgba(255,255,255,.7));
    border-inline-start:5px solid var(--teal); padding:18px 20px; border-radius:14px; margin-top:18px;
    color:var(--muted); font-size:14.5px; line-height:1.75; }
.info-box strong { color:var(--navy); display:block; margin-bottom:4px; font-size:15.5px; }

/* ═════════ Footer ═════════ */
.foot { text-align:center; margin-top:46px; padding-top:22px; border-top:1px solid var(--line); color:#7E95A3; font-size:13px; font-weight:500; }
.credit { margin-top:16px; text-align:center; color:#7E95A3; font-size:14px; font-weight:500; }
.credit .heart { display:inline-block; color:var(--coral); animation:beat 1.3s ease-in-out infinite; }
@keyframes beat { 0%,100% { transform:scale(1); } 15% { transform:scale(1.35); } 30% { transform:scale(1); } 45% { transform:scale(1.25); } }
.credit b { color:var(--teal); font-weight:800; letter-spacing:1px; }

.stAlert { border-radius:14px; font-weight:600; }

@media (max-width:760px) {
    .hero-title { font-size:40px; }
    .result-card { flex-direction:column; text-align:center; }
}
@media (prefers-reduced-motion:reduce) {
    *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
    .bubble { display:none; }
}
</style>
"""
CSS = CSS.replace("__FD__", font_display).replace("__FB__", font_body).replace("__DIR__", direction)


def html(block: str):
    """Render an HTML block. Blank lines / indentation are stripped so markdown never turns it into a code block."""
    cleaned = "\n".join(line.strip() for line in block.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


html(CSS)

# ──────────────────────────────────────────────────────────────
#  SVG assets
# ──────────────────────────────────────────────────────────────
LUNG_L = "M28 18 C19 21 11 34 11 47 C11 53 16 55 21 52 C26 49 29 44 29 37 Z"
LUNG_R = "M36 18 C45 21 53 34 53 47 C53 53 48 55 43 52 C38 49 35 44 35 37 Z"
AIRWAY = "M32 9 V29 M32 29 C32 33 29 34 25 38 M32 29 C32 33 35 34 39 38"

LOGO_SVG = f"""
<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PneumoScan AI logo">
<defs>
<linearGradient id="lg-bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#14A5A8"/><stop offset="1" stop-color="#0B2A4A"/></linearGradient>
<linearGradient id="lg-lung" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#BFEFF0"/></linearGradient>
</defs>
<rect width="64" height="64" rx="19" fill="url(#lg-bg)"/>
<rect x="1.5" y="1.5" width="61" height="61" rx="17.5" fill="none" stroke="rgba(255,255,255,.28)" stroke-width="1.2"/>
<path class="lung l" d="{LUNG_L}" fill="url(#lg-lung)"/>
<path class="lung r" d="{LUNG_R}" fill="url(#lg-lung)"/>
<path d="{AIRWAY}" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
<path class="ecg-mini" d="M4 42 H17 L22 34 L27 49 L33 30 L38 42 H60" fill="none" stroke="#0B2A4A" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" opacity=".35"/>
<path class="ecg-mini" d="M4 42 H17 L22 34 L27 49 L33 30 L38 42 H60" fill="none" stroke="#5CF2E4" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ECG_SVG = """
<svg class="ecg" viewBox="0 0 520 56" preserveAspectRatio="none" aria-hidden="true">
<defs><linearGradient id="ecg-g" x1="0" x2="1"><stop offset="0" stop-color="#3CCFCF" stop-opacity="0"/><stop offset=".25" stop-color="#3CCFCF"/><stop offset=".75" stop-color="#0F8B8D"/><stop offset="1" stop-color="#0F8B8D" stop-opacity="0"/></linearGradient></defs>
<path class="base" pathLength="1000" fill="none" stroke-width="2" d="M0 30 H150 L165 30 L175 12 L190 52 L205 5 L218 30 H330 L342 30 L350 19 L360 38 L368 30 H520"/>
<path class="trace" pathLength="1000" fill="none" stroke="url(#ecg-g)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" d="M0 30 H150 L165 30 L175 12 L190 52 L205 5 L218 30 H330 L342 30 L350 19 L360 38 L368 30 H520"/>
</svg>
"""

ICON_EYE = '<svg viewBox="0 0 24 24"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/><circle cx="12" cy="12" r="3"/></svg>'
ICON_BOLT = '<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>'
ICON_FILE = '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M9 14h6M9 18h6"/></svg>'
ICON_USER = '<svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
ICON_CHECK = '<svg viewBox="0 0 24 24"><path d="M4 12.5 9.5 18 20 6"/></svg>'
ICON_ALERT = '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20z"/><path d="M12 10v5M12 18v.5"/></svg>'

# (left %, size px, duration s, delay s)
BUBBLES = [
    (4, 26, 19, 0), (11, 54, 27, 6), (19, 18, 16, 3), (27, 38, 23, 11), (36, 70, 31, 2),
    (45, 22, 18, 8), (53, 46, 25, 14), (61, 16, 15, 5), (69, 60, 29, 9), (77, 30, 21, 1),
    (84, 20, 17, 12), (91, 48, 26, 4), (96, 28, 20, 16),
]
bubbles_html = "".join(
    f'<div class="bubble" style="left:{l}%;width:{s}px;height:{s}px;animation-duration:{d}s;animation-delay:{dl}s"></div>'
    for l, s, d, dl in BUBBLES
)
html(f'<div class="bg-blob b1"></div><div class="bg-blob b2"></div>{bubbles_html}')


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        return model, True
    return None, False


def generate_mock_cam(image):
    """Soft red/yellow blob over the X-ray, used only when the model file is missing."""
    base = image.resize((224, 224)).convert("RGB")
    mask = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = random.randint(70, 150), random.randint(80, 150)
    rad = random.randint(24, 46)
    draw.ellipse([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(22))
    mask = mask.point(lambda p: int(p * 0.75))
    heat = Image.new("RGB", base.size, (220, 38, 38))
    return np.array(Image.composite(heat, base, mask))


def to_b64(img, max_w=720):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype("uint8"))
    img = img.convert("RGB")
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def scan_frame(img, caption, slow=False, legend=False):
    legend_html = (
        f'<div class="legend"><span>{t["legend_low"]}</span><div class="bar"></div><span>{t["legend_high"]}</span></div>'
        if legend else ""
    )
    html(f"""
    <div class="scan-wrap" dir="{direction}">
        <div class="scan-frame{' slow' if slow else ''}">
            <img src="data:image/jpeg;base64,{to_b64(img)}" alt="{caption}">
            <span class="corner tl"></span><span class="corner tr"></span>
            <span class="corner bl"></span><span class="corner br"></span>
            <div class="scan-line"></div>
        </div>
        <div class="img-caption">{caption}</div>
        {legend_html}
    </div>
    """)


def build_pdf_report(doctor, patient_name, patient_id, original_img, cam_array, label, confidence):
    T = PDF_T
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    teal = rl_colors.HexColor("#0F8B8D")
    navy = rl_colors.HexColor("#0B2A4A")

    title_style = ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20,
                                 textColor=navy, spaceAfter=6)
    sub_style = ParagraphStyle("s", parent=styles["Normal"], fontName="Helvetica", fontSize=9,
                               textColor=rl_colors.HexColor("#5B7083"), spaceAfter=16)
    h_style = ParagraphStyle("h", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                             textColor=teal, spaceBefore=14, spaceAfter=8)
    normal_style = ParagraphStyle("n", parent=styles["Normal"], fontName="Helvetica", fontSize=10,
                                  textColor=rl_colors.HexColor("#16324F"), leading=15)

    story = [
        Paragraph(T["report_title"], title_style),
        Paragraph(f"{T['field_date']}: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", sub_style),
        Paragraph(T["form_title"], h_style),
    ]

    patient_data = [
        [T["field_patient"], patient_name or "-"],
        [T["field_patient_id"], patient_id or "-"],
        [T["field_doctor"], doctor or "-"],
    ]
    pt = Table(patient_data, colWidths=[5 * cm, 11 * cm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#EAF4F6")),
        ("TEXTCOLOR", (0, 0), (0, -1), navy),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#D5E5EA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pt)

    story.append(Paragraph(T["prediction"], h_style))
    res_color = "#C8262C" if label == "PNEUMONIA" else "#15803D"
    res_style = ParagraphStyle("r", parent=normal_style, fontSize=14,
                               textColor=rl_colors.HexColor(res_color), spaceAfter=6)
    display_label = T["pneumonia"] if label == "PNEUMONIA" else T["normal"]
    story.append(Paragraph(f"<b>{T['field_prediction']}:</b> {display_label}", res_style))
    story.append(Paragraph(f"<b>{T['field_confidence']}:</b> {confidence:.2f}%", normal_style))

    story.append(Paragraph(T["heatmap"], h_style))

    orig_buf = io.BytesIO()
    original_img.resize((280, 280)).save(orig_buf, format="PNG")
    orig_buf.seek(0)

    cam_buf = io.BytesIO()
    Image.fromarray(cam_array).resize((280, 280)).save(cam_buf, format="PNG")
    cam_buf.seek(0)

    imgs = Table([
        [RLImage(orig_buf, width=7 * cm, height=7 * cm), RLImage(cam_buf, width=7 * cm, height=7 * cm)],
        [Paragraph(T["field_original"], normal_style), Paragraph(T["field_heatmap"], normal_style)],
    ], colWidths=[8 * cm, 8 * cm])
    imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, 1), 9),
    ]))
    story.append(imgs)

    story.append(Paragraph(T["disclaimer_title"], h_style))
    story.append(Paragraph(T["report_disclaimer_text"], normal_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ──────────────────────────────────────────────────────────────
#  Language toggle
# ──────────────────────────────────────────────────────────────
_, top_right = st.columns([8, 2])
with top_right:
    if st.button(t["lang_btn"], key="lang_toggle", **STRETCH):
        st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
        st.rerun()

# ──────────────────────────────────────────────────────────────
#  Hero
# ──────────────────────────────────────────────────────────────
html(f"""
<div class="hero" dir="{direction}">
    <div class="logo-badge">{LOGO_SVG}</div>
    <div class="hero-tag"><span class="live-dot"></span>{t['tag']}</div>
    <div class="hero-title">Pneumo<span class="scan">Scan</span><span class="ai">AI</span></div>
    <div class="hero-sub">{t['subtitle']}</div>
    {ECG_SVG}
    <div class="hero-desc">{t['desc']}</div>
    <div class="feat-row">
        <span class="feat">{ICON_EYE}{t['feat1']}</span>
        <span class="feat">{ICON_BOLT}{t['feat2']}</span>
        <span class="feat">{ICON_FILE}{t['feat3']}</span>
    </div>
</div>
""")

# ──────────────────────────────────────────────────────────────
#  Main layout
# ──────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    with st.container(border=True):
        html(f'<div class="form-title" dir="{direction}"><span class="ico">{ICON_USER}</span>{t["form_title"]}</div>')

        doctor_name = st.text_input(t["doctor_name"], placeholder="Dr. Smith")
        patient_name = st.text_input(t["patient_name"], placeholder="John Doe")
        patient_id = st.text_input(t["patient_id"], placeholder="P-1029")

        st.markdown(f"**{t['upload_label']}**")
        uploaded_file = st.file_uploader(
            t["upload_label"], type=["jpg", "jpeg", "png"], label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        html(f'<div class="divider-or">{t["or"]}</div>')
        demo_clicked = st.button(t["demo_btn"], **STRETCH)

        # Only react to a *new* file, otherwise every rerun (download, language switch...) wiped the result.
        if uploaded_file is not None:
            sig = (uploaded_file.name, uploaded_file.size)
            if st.session_state.last_sig != sig:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.analysis_result = None
                st.session_state.last_sig = sig
                st.toast(t["status_uploaded"], icon="📤")
                
        if demo_clicked:
            with st.spinner("جاري جلب صورة طبية..." if is_rtl else "Fetching medical image..."):
                try:
                    import requests
                    offset = random.randint(0, 600)
                    api_url = f"https://datasets-server.huggingface.co/rows?dataset=keremberke%2Fchest-xray-classification&config=full&split=test&offset={offset}&length=1"
                    res = requests.get(api_url, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        img_src = data['rows'][0]['row']['image']['src']
                        img_res = requests.get(img_src, timeout=15)
                        if img_res.status_code == 200:
                            st.session_state.uploaded_image = Image.open(io.BytesIO(img_res.content)).convert("RGB")
                            st.session_state.analysis_result = None
                            st.toast(t["status_demo"], icon="🎲")
                        else:
                            st.error("Failed to download the image file.")
                    else:
                        st.error("Failed to connect to the medical image database.")
                except Exception as e:
                    st.error(f"Error fetching image: {e}")


        if st.session_state.uploaded_image is not None:
            scan_frame(st.session_state.uploaded_image, t["original"])

            if st.button(t["analyze_btn"], **STRETCH):
                loader = st.empty()
                with loader:
                    html(f"""
                    <div class="loader" dir="{direction}">
                        <div class="loader-ring"></div>
                        <div class="txt">{t['analyzing']} <span class="dots"><span></span><span></span><span></span></span></div>
                    </div>
                    """)
                started = time.time()
                warn_missing_model = False

                model, is_real = load_model()

                if is_real:
                    inference_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    img_tensor = inference_transform(st.session_state.uploaded_image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        predicted_class = torch.argmax(probabilities).item()
                        confidence = probabilities[predicted_class].item() * 100

                    target_layers = [model.features[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(predicted_class)]
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
                    rgb_img = np.array(st.session_state.uploaded_image.resize((224, 224))) / 255.0
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    label = CLASS_NAMES[predicted_class]
                else:
                    label = random.choice(CLASS_NAMES)
                    confidence = random.uniform(85.0, 98.5)
                    cam_image = generate_mock_cam(st.session_state.uploaded_image)
                    warn_missing_model = True

                # keep the loader on screen long enough to be seen
                time.sleep(max(0.0, 1.0 - (time.time() - started)))
                loader.empty()

                if warn_missing_model:
                    st.warning(t["missing_model"])

                st.session_state.analysis_result = {
                    "label": label,
                    "confidence": confidence,
                    "cam_image": cam_image,
                    "doctor": doctor_name,
                    "patient": patient_name,
                    "patient_id": patient_id,
                }

with col_result:
    with st.container(border=True):
        if st.session_state.analysis_result is None:
            html(f"""
            <div class="empty" dir="{direction}">
                <div class="orb">
                    <span class="ripple"></span><span class="ripple"></span><span class="ripple"></span>
                    <svg viewBox="0 0 64 64" aria-hidden="true">
                        <path class="lung-outline" d="{LUNG_L}"/>
                        <path class="lung-outline" d="{LUNG_R}"/>
                        <path d="{AIRWAY}" fill="none" stroke="#0F8B8D" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <h3>{t['status_ready']}</h3>
                <p>{t['status_ready_sub']}</p>
            </div>
            """)
        else:
            result = st.session_state.analysis_result
            is_pneumonia = result["label"] == "PNEUMONIA"
            res_class = "res-pneumonia" if is_pneumonia else "res-normal"
            val_class = "val-pneumonia" if is_pneumonia else "val-normal"
            display_label = t["pneumonia"] if is_pneumonia else t["normal"]
            status_icon = ICON_ALERT if is_pneumonia else ICON_CHECK

            conf = round(result["confidence"], 1)
            conf_int = int(conf)
            conf_dec = int(round((conf - conf_int) * 10))
            ring_offset = 326.7 * (1 - conf / 100)

            html(f"""
            <div class="result-card {res_class}" dir="{direction}">
                <div class="ring-wrap">
                    <svg class="ring" viewBox="0 0 120 120">
                        <circle class="ring-bg" cx="60" cy="60" r="52"/>
                        <circle class="ring-fg" cx="60" cy="60" r="52" style="--off:{ring_offset:.1f}"/>
                    </svg>
                    <div class="ring-num"><span class="count" style="--target:{conf_int}"></span><small>.{conf_dec}%</small></div>
                </div>
                <div class="res-text">
                    <div class="res-label">{t['prediction']}</div>
                    <div class="res-value {val_class}">{display_label}</div>
                    <div class="res-conf">{status_icon}{t['confidence']}: {conf:.1f}%</div>
                </div>
            </div>
            """)

            scan_frame(result["cam_image"], t["heatmap"], slow=True, legend=True)

            html(f"""
            <div class="info-box" dir="{direction}">
                <strong>{t['howto_title']}</strong>
                {t['howto_desc']}
            </div>
            """)

            st.markdown("<br>", unsafe_allow_html=True)

            pdf_buffer = build_pdf_report(
                result["doctor"], result["patient"], result["patient_id"],
                st.session_state.uploaded_image, result["cam_image"],
                result["label"], result["confidence"],
            )

            st.download_button(
                label=t["download_report"],
                data=pdf_buffer,
                file_name=f"PneumoScan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                **STRETCH,
            )

            if st.button(t["reanalyze_btn"], key="reanalyze_button", **STRETCH):
                st.session_state.analysis_result = None
                st.session_state.uploaded_image = None
                st.session_state.last_sig = None
                st.session_state.uploader_key += 1  # resets the file uploader widget
                st.rerun()

# ──────────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────────
html(f'<div class="foot">{t["disclaimer"]}</div>')
html(f"""
<div class="credit" dir="{direction}">
    {t['made_with']} <span class="heart">❤</span> {t['by']} <b>SHODO</b>
</div>
""")
