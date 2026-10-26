<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=32&pause=1000&color=0A6EBD&center=true&vCenter=true&width=800&lines=PneumoScan+AI;Medical+Image+Analysis;Explainable+AI+(Grad-CAM);Full+Bilingual+(RTL)+Support" alt="Typing SVG" />

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Stethoscope.png" alt="Stethoscope" width="50" />
</div>

<div align="center">
  <p>🩵 ˚ ༘ ೀ⋆｡ ˚ 🏥 ˚ ೀ⋆｡ ˚ 🩵</p>
</div>

<br>

**𝐏𝐧𝐞𝐮𝐦𝐨𝐒𝐜𝐚𝐧 𝐀𝐈** ⛥

*𝑻𝒓𝒂𝒏𝒔𝒇𝒐𝒓𝒎𝒊𝒏𝒈 𝑹𝒂𝒘 𝑴𝒆𝒅𝒊𝒄𝒂𝒍 𝑰𝒎𝒂𝒈𝒆𝒔 𝒊𝒏𝒕𝒐 𝑳𝒊𝒇𝒆-𝑺𝒂𝒗𝒊𝒏𝒈 𝑫𝒊𝒂𝒈𝒏𝒐𝒔𝒆𝒔*

<br>

[![Live Demo](https://img.shields.io/badge/🩺_LIVE_DEMO-TRY_IT_NOW-success?style=for-the-badge&logo=streamlit&logoColor=white)](https://pneumoscan-ai-shodo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

> [!NOTE]💡
>
> الرابط أعلاه هو تطبيق حي ومباشر (Live App) يمكنك تجربته فوراً من متصفحك دون الحاجة لتحميل أي شيء.
>
> **لتشغيل المشروع على جهازك الخاص، يمكنك اتباع الخطوات الموضحة أدناه.**
>
> *The link above is a fully deployed Live App. You can test it instantly in your browser. To run it locally on your machine, follow the steps below.*

---

## ⚡ 𝑷𝒓𝒐𝒋𝒆𝒄𝒕 𝑶𝒗𝒆𝒓𝒗𝒊𝒆𝒘 | نظرة عامة

**PneumoScan AI** is a premium, interactive clinical dashboard designed to assist in detecting Pneumonia from Chest X-Ray images. It features a sleek medical-grade design, dynamic visualizations, bilingual support (English/Arabic RTL), and explainable AI.

The core of the project is a **Deep Learning Application** powered by **PyTorch (MobileNetV3)** and **Grad-CAM** to not only provide a diagnosis but also explain *where* the AI is looking.

---

## 🚀 3 Steps to Run Locally | خطوات تشغيل المشروع محلياً

To run the complete interactive project on your machine, follow these three simple steps:

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/Shahad58/PneumoScanAI.git
cd PneumoScanAI
```

**Step 2 — Install dependencies:**

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

**Step 3 — Run the application:**

```bash
python -m streamlit run app.py
```

> 🎉 **Success!** The app will now be live at `http://localhost:8501` in your browser.
> *(Note: Ensure the `pneumonia_mobilenet.pth` file is in the root directory for instant AI inference).*

---

## ✨ 𝑲𝒆𝒚 𝑭𝒆𝒂𝒕𝒖𝒓𝒆𝒔 | المميزات الرئيسية

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Explainable AI (Grad-CAM)</h3>
      <p>Uses <b>Grad-CAM</b> to generate a heatmap, highlighting the exact regions of the lung the AI focused on to make its diagnosis.</p>
    </td>
    <td width="50%">
      <h3>🤖 Deep Learning Diagnostics</h3>
      <p>Leverages a fine-tuned <b>MobileNetV3</b> architecture via Transfer Learning on the Kaggle Chest X-Ray dataset for fast and accurate detection.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📄 Clinical PDF Reports</h3>
      <p>Generates downloadable, professional medical reports including patient details, original X-ray, AI heatmap, and confidence scores using <b>ReportLab</b>.</p>
    </td>
    <td width="50%">
      <h3>🌍 True Bilingual UI (RTL)</h3>
      <p>Seamless switching between English and Arabic. The entire layout, fonts, and text directions adapt automatically to Right-to-Left.</p>
    </td>
  </tr>
</table>

---

## 📸 𝑫𝒂𝒔𝒉𝒃𝒐𝒂𝒓𝒅 𝑷𝒓𝒆𝒗𝒊𝒆𝒘 | معاينة الواجهة

<div align="center">
  <br>
  


  
  <br>
</div>

---

## 🛠️ 𝑻𝒆𝒄𝒉 𝑺𝒕𝒂𝒄𝒌 | التقنيات المستخدمة

<div align="center">

| Category | Technologies |
| :--- | :--- |
| **Frontend / UI** | `Streamlit`, `HTML5`, `CSS3`, `Glassmorphism` |
| **Backend** | `Python 3.11` |
| **Deep Learning** | `PyTorch`, `Torchvision`, `MobileNetV3` |
| **Computer Vision** | `OpenCV`, `PIL`, `Grad-CAM` |
| **Reporting** | `ReportLab (PDF Generation)` |
| **Design** | `Medical Light Theme`, `Animated UI`, `RTL Support` |

</div>

---

## 📁 𝑷𝒓𝒐𝒋𝒆𝒄𝒕 𝑺𝒕𝒓𝒖𝒄𝒕𝒖𝒓𝒆 | هيكل المشروع

```
/PneumoScanAI
├── app.py                     ← Full application (Streamlit + AI + UI)
├── requirements.txt           ← Python dependencies
├── packages.txt               ← System dependencies (libgl1 for OpenCV)
├── pneumonia_mobilenet.pth    ← Pre-trained PyTorch Model Weights
└── README.md                  ← This file
```

> **ملاحظة:** عند التجربة، يعمل النموذج فوراً دون الحاجة لتدريب. تأكد من وجود ملف الأوزان `pneumonia_mobilenet.pth` في نفس المجلد.

---

<div align="center">

## 👨‍💻 𝑨𝒖𝒕𝒉𝒐𝒓 | المطور

**SHODO** ⛥

[![GitHub Follow](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shahad58)

<br>
<i>"Turning Medical Images Into Life-Saving Decisions."</i>

<br><br>

**Created with 🩵 by SHODO**

</div>
