# AdamAI-Systems — Medical

Medical imaging, diagnosis, and healthcare AI applications. This repository contains deep learning projects focused on **radiology**, **medical image analysis**, and **clinical AI** developed by **AdamAI-Systems**.

## Projects in this Repository

### 1. [AI Radiologist — Chest X-Ray Pneumonia Detection](./ai-radiologist-gradio)
A transfer learning model that classifies chest X-ray images as **Normal** or **Pneumonia**, complete with an interactive **Gradio** web interface for real-time diagnosis.
* **Tech Stack:** Python, TensorFlow / Keras, MobileNetV2, Gradio, HuggingFace Datasets.
* **Key Features:** Transfer learning with MobileNetV2 backbone, automated dataset download, interactive visual diagnosis interface.

### 2. [RSNA Pneumonia Detection — Multi-Task Classification & Localization](./rsna-pneumonia-cls-det)
A multi-task deep learning model trained on the **RSNA Pneumonia Detection Challenge** dataset that simultaneously classifies pneumonia presence and localizes infection regions via bounding boxes.
* **Tech Stack:** Python, PyTorch, pydicom, kagglehub, scikit-learn.
* **Key Features:** DICOM image reading and preprocessing, custom CNN with classification + grid-based detection heads, NMS-based box decoding, ROC-AUC and IoU evaluation.

---

## Setup & General Instructions

Each project runs inside a **Jupyter Notebook** (designed for Google Colab). Navigate to the project directory and follow the instructions in its `README.md`:

- Chest X-Ray Classifier: [ai-radiologist-gradio/README.md](./ai-radiologist-gradio/README.md)
- RSNA Multi-Task Detector: [rsna-pneumonia-cls-det/README.md](./rsna-pneumonia-cls-det/README.md)

> **Note:** Model weights (`.h5`, `.pth`) and datasets are excluded from this repository via `.gitignore`. Run the notebooks to download data and train the models.

---
© 2026 AdamAI-Systems. All rights reserved.
