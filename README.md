<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F8B8D,100:0B2A4A&height=210&section=header&text=PneumoScan%20AI&fontSize=54&fontColor=ffffff&fontAlignY=38&desc=Explainable%20Pneumonia%20Detection%20from%20Chest%20X-Rays&descSize=18&descAlignY=60&animation=fadeIn" alt="PneumoScan AI banner" width="100%"/>

<img src="https://readme-typing-svg.herokuapp.com?font=Sora&weight=600&size=24&pause=1200&color=0F8B8D&center=true&vCenter=true&width=760&lines=Medical+Image+Classification;MobileNetV3+%2B+Transfer+Learning;Explainable+AI+with+Grad-CAM;Bilingual+Arabic+%2F+English+Interface;One-click+Clinical+PDF+Reports" alt="Typing SVG" />

<br><br>

[![Live Demo](https://img.shields.io/badge/🩺_Live_Demo-Try_it_now-0F8B8D?style=for-the-badge&logo=streamlit&logoColor=white)](https://pneumoscan-ai-shodo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## 🇸🇦 نظرة عامة

**PneumoScan AI** تطبيق ويب متكامل للكشف عن الالتهاب الرئوي من صور الأشعة السينية للصدر. يعتمد على نموذج تعلّم عميق (**MobileNetV3**) مبني بواسطة **PyTorch** ومدرَّب بأسلوب *Transfer Learning*، ويقدّم واجهة تفاعلية حديثة باللغتين العربية والإنجليزية.

الميزة الأهم أن النظام لا يعطيك نتيجة فقط، بل **يشرح قراره** عبر خريطة حرارية (Grad-CAM) تُظهر المناطق التي ركّز عليها النموذج، مما يرفع الثقة في النتيجة ويسهّل مراجعتها.

## 🇬🇧 Overview

**PneumoScan AI** is a full web application that detects pneumonia from chest X-ray images. It runs a **MobileNetV3** classifier built with **PyTorch** and fine-tuned with transfer learning, wrapped in a modern, interactive **Streamlit** interface available in both Arabic and English.

Beyond a label and a confidence score, PneumoScan **explains its decision**: a Grad-CAM heatmap highlights the lung regions that influenced the prediction, making the output easier to review and trust.

---

## ✨ Features

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🧠 Explainable AI · تفسير القرار</h3>
      <p>Grad-CAM heatmaps show which regions drove the prediction.<br>
      <sub>خريطة حرارية توضح المناطق التي اعتمد عليها النموذج.</sub></p>
    </td>
    <td width="50%" valign="top">
      <h3>⚡ Instant Diagnosis · تشخيص فوري</h3>
      <p>Normal / Pneumonia classification with a confidence score.<br>
      <sub>تصنيف الصورة مع نسبة الثقة في أقل من ثانية.</sub></p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>📄 Clinical PDF Reports · تقارير طبية</h3>
      <p>Export a report with patient data, the original X-ray, and the heatmap.<br>
      <sub>تقرير PDF يضم بيانات المريض والصورة والخريطة الحرارية.</sub></p>
    </td>
    <td width="50%" valign="top">
      <h3>🎲 Demo Mode · تجربة سريعة</h3>
      <p>Load a random sample X-ray in one click, no upload required.<br>
      <sub>حمّل صورة أشعة عشوائية بضغطة واحدة بدون رفع ملف.</sub></p>
    </td>
  </tr>
</table>

---

## 📸 App Preview

<div align="center">
  <!-- ضع رابط صورتك هنا بدلاً من هذا النص -->
  <img src="https://via.placeholder.com/1000x520/EAF4F6/0F8B8D?text=PneumoScan+AI+-+Dashboard+Preview" alt="Home screen" width="100%"/>
</div>

---

## 🔬 How It Works

```mermaid
flowchart LR
    A[🩻 Chest X-ray] --> B[Pre-processing<br/>224×224]
    B --> C[MobileNetV3<br/>Classifier]
    C --> D{Normal / Pneumonia}
    C --> G[Grad-CAM<br/>Heatmap]
    D --> I[📄 PDF Report]
    G --> I
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning** | `PyTorch` · `Torchvision` · `MobileNetV3` (transfer learning) |
| **Explainable AI** | `pytorch-grad-cam` (Grad-CAM) |
| **Web App & UI** | `Python 3.11` · `Streamlit` · custom CSS / SVG |
| **Image Processing** | `Pillow` · `NumPy` · `OpenCV` |
| **Reporting** | `ReportLab` |

</div>

---

## 📂 Project Structure & Live Demo

```text
PneumoScanAI/
├── app.py                    # Streamlit app (UI, inference, Grad-CAM, PDF)
├── pneumonia_mobilenet.pth   # Trained model weights
├── requirements.txt          # Python dependencies
└── README.md
```

👉 **تجربة التطبيق مباشرة | Try the App Now:** 
[**انقر هنا لتجربة الموقع (Click Here)**](https://pneumoscan-ai-shodo.streamlit.app/)

---

## 🚀 Getting Started

**Prerequisites:** Python 3.11+ and `pip`. A GPU is optional.

**1. Clone the repository**
```bash
git clone https://github.com/Shahad58/PneumoScanAI.git
cd PneumoScanAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add model weights**
Ensure `pneumonia_mobilenet.pth` is in the project root.

**4. Run the app**
```bash
streamlit run app.py
```

---

## ⚠️ Disclaimer

> **EN:** PneumoScan AI is built for **educational purposes only**. It is not a medical device and must not replace diagnosis by a qualified radiologist.
>
> **AR:** هذا المشروع مخصص **للأغراض التعليمية فقط**، وليس جهازاً طبياً، ولا يغني عن تشخيص طبيب مختص.

---

<div align="center">

## 👨‍💻 Author

**SHODO** ⛥

[![GitHub](https://img.shields.io/badge/GitHub-Shahad58-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shahad58)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B2A4A,100:0F8B8D&height=110&section=footer" width="100%" alt=""/>
</div>
