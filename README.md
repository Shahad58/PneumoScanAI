<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=32&pause=1000&color=0A6EBD&center=true&vCenter=true&width=800&lines=PneumoScan+AI;Medical+Image+Classification;Explainable+AI+(Grad-CAM);Streamlit+Deployment" alt="Typing SVG" />

<br>

[![Live Demo](https://img.shields.io/badge/🩺_LIVE_DEMO-TRY_IT_NOW-success?style=for-the-badge&logo=streamlit&logoColor=white)](https://pneumoscan-ai-shodo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## 🇸🇦 نظرة عامة

**PneumoScan AI** هو تطبيق ويب للكشف عن الالتهاب الرئوي من صور الأشعة السينية (Chest X-Rays). يعتمد النظام على نموذج تعلم عميق (MobileNetV3) مبني بواسطة PyTorch، ويوفر واجهة تفاعلية باللغتين العربية والإنجليزية.

**المميزات:**
* **تشخيص فوري:** تصنيف الصورة (سليمة / التهاب رئوي) مع حساب نسبة الثقة.
* **تفسير القرار (Grad-CAM):** توليد خريطة حرارية توضح للطبيب المناطق التي اعتمد عليها النموذج في التشخيص.
* **تقارير طبية:** إمكانية تحميل تقرير PDF يحتوي على بيانات المريض، الصورة، والنتائج.
* **بيانات تجريبية ديناميكية:** جلب صور أشعة حقيقية عشوائية من قواعد بيانات مفتوحة المصدر لتجربة التطبيق فوراً.
* **دعم RTL:** واجهة مستخدم متجاوبة مع دعم كامل للغة العربية.

---

## 🇬🇧 Overview

**PneumoScan AI** is a web application for detecting Pneumonia from Chest X-Rays. The system utilizes a Deep Learning model (MobileNetV3) built with PyTorch, providing an interactive interface in both Arabic and English.

**Features:**
* **Instant Diagnosis:** Classifies X-rays (Normal / Pneumonia) with a confidence score.
* **Explainable AI (Grad-CAM):** Generates heatmaps to visualize the regions influencing the model's decision.
* **Clinical Reports:** Export detailed PDF reports including patient data, original X-ray, and AI heatmap.
* **Dynamic Demo Mode:** Fetches random, real X-rays from an open-source dataset for instant testing without manual uploads.
* **RTL Support:** Fully responsive UI with native Right-to-Left Arabic support.

---

## 📸 لقطات من التطبيق | App Preview

<div align="center">
  <br>
  
  <!-- ضع رابط صورتك هنا بدلاً من هذا النص -->
  <img width="100%" alt="PneumoScan UI" src="https://via.placeholder.com/1000x500/F4F7FB/0A6EBD?text=PneumoScan+AI+Dashboard" />
  
  <br>
</div>

---

## 🛠️ التقنيات | Tech Stack

<div align="center">

| Field | Technologies |
| :--- | :--- |
| **Deep Learning** | `PyTorch`, `Torchvision`, `MobileNetV3` (Transfer Learning) |
| **Explainable AI** | `Grad-CAM` (Custom implementation) |
| **Backend & UI** | `Python 3.11`, `Streamlit` |
| **Computer Vision** | `PIL`, `OpenCV` |
| **Reporting** | `ReportLab` (PDF Generation) |

</div>

---

## 🚀 التشغيل المحلي | How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/Shahad58/PneumoScanAI.git
cd PneumoScanAI
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

> **Note:** Ensure the pre-trained model weights (`pneumonia_mobilenet.pth`) are placed in the root directory before running.

---

<div align="center">

## 👨‍💻 المطور | Author

**SHODO** ⛥

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Shahad58)

</div>
