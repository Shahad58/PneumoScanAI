<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=32&pause=1000&color=0A6EBD&center=true&vCenter=true&width=800&lines=PneumoScan+AI;Medical+Image+Classification;Explainable+AI+(Grad-CAM);Streamlit+Web+App" alt="Typing SVG" />

<br>

[![Live Demo](https://img.shields.io/badge/🩺_LIVE_DEMO-TRY_IT_NOW-success?style=for-the-badge&logo=streamlit&logoColor=white)](https://pneumoscan-ai-shodo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## 🇸🇦 نظرة عامة

**PneumoScan AI** هو تطبيق ويب متكامل للكشف عن الالتهاب الرئوي من صور الأشعة السينية. يعتمد النظام على نموذج تعلم عميق (MobileNetV3) مبني بواسطة PyTorch، ويقدم واجهة تفاعلية باللغتين العربية والإنجليزية.

### ✨ المميزات الرئيسية

<table>
  <tr>
    <td width="50%">
      <h3>🧠 تفسير القرار (Grad-CAM)</h3>
      <p>توليد خريطة حرارية توضح للطبيب المناطق التي اعتمد عليها النموذج في التشخيص، مما يعزز الثقة في النتائج.</p>
    </td>
    <td width="50%">
      <h3>⚡ تشخيص فوري</h3>
      <p>تصنيف الصورة (سليمة / التهاب) مع حساب نسبة الثقة بدقة، ليعطي نتيجة في أقل من ثانية.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📄 تقارير طبية (PDF)</h3>
      <p>إمكانية تحميل تقرير فني شامل يحتوي على بيانات المريض، الصورة الأصلية، الخريطة الحرارية، والنتيجة.</p>
    </td>
    <td width="50%">
      <h3>🎲 تجربة ديناميكية</h3>
      <p>جلب صور أشعة حقيقية وعشوائية من قواعد بيانات مفتوحة المصدر (Hugging Face) لتجربة النظام فوراً.</p>
    </td>
  </tr>
</table>

---

## 🇬🇧 Overview

**PneumoScan AI** is a comprehensive web application designed to detect Pneumonia from Chest X-Rays. The system utilizes a Deep Learning model (MobileNetV3) built with PyTorch, providing an interactive interface in both Arabic and English.

### ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Explainable AI (Grad-CAM)</h3>
      <p>Generates heatmaps to visualize the regions influencing the model's decision, enhancing clinical trust and interpretability.</p>
    </td>
    <td width="50%">
      <h3>⚡ Instant Diagnosis</h3>
      <p>Classifies X-rays (Normal / Pneumonia) with high accuracy and a confidence score, delivering results in under a second.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📄 Clinical PDF Reports</h3>
      <p>Export detailed technical reports including patient data, original X-ray, AI heatmap, and final diagnosis.</p>
    </td>
    <td width="50%">
      <h3>🎲 Dynamic Demo Mode</h3>
      <p>Fetches random, real X-rays from open-source datasets (Hugging Face) for instant testing without manual uploads.</p>
    </td>
  </tr>
</table>

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
| **Explainable AI** | `Grad-CAM` |
| **Backend & UI** | `Python 3.11`, `Streamlit` |
| **Computer Vision** | `PIL`, `OpenCV` |
| **Reporting** | `ReportLab` |

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

> **Note:** Ensure the pre-trained weights (`pneumonia_mobilenet.pth`) are in the root directory before running.

---

<div align="center">

## 👨‍💻 المطور | Author

**SHODO** ⛥

[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Shahad58)

</div>
