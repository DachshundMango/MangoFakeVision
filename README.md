# 🥭 MangoFakeVision 🚀  
**Deepfake Detection with CNN, Transformers, and Metadata Analysis**  

MangoFakeVision is an advanced deepfake detection system using **CNN (ResNet, EfficientNet), Vision Transformer (ViT, Swin-T), and metadata analysis** to identify manipulated images and videos.

---

## 📌 Features  
✅ **Multi-Model Approach** – CNN & Transformer-based detection  
✅ **Data Augmentation** – `torchvision.transforms` (CNN) & **Stable Diffusion / GAN** (Transformers)  
✅ **Metadata Extraction** – EXIF, codec, compression analysis  
✅ **Real-Time Detection** – FastAPI & Streamlit Web UI  

---

## 📂 Project Structure  
```plaintext
MangoFakeVision/
│── data/          # Dataset (Raw, Processed, Augmented, Metadata)
│── models/        # Trained models (CNN, Transformer, ONNX)
│── scripts/       # Training, Augmentation, Metadata Extraction
│── deployment/    # FastAPI API & Streamlit Web UI
│── README.md      # Project Documentation
```

---

## 🚀 Quick Start  
### **1️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
### **2️⃣ Download & Process Dataset**  
```bash
python scripts/preprocess.py
```
### **3️⃣ Train Models**  
```bash
python scripts/train_cnn.py    # Train CNN  
python scripts/train_transformer.py  # Train Transformer  
```
### **4️⃣ Run API & Web UI**  
```bash
python deployment/app.py       # FastAPI Server  
streamlit run frontend/app.py  # Streamlit UI  
```

---

## 📊 Model Performance  
| Model       | Accuracy | F1-Score |
|------------|----------|----------|
| ResNet-50  | -        | -        |
| EfficientNet | -      | -        |
| ViT-Base   | -        | -        |
| Swin-T     | -        | -        |

---

## 🔍 Metadata Analysis  
Extract EXIF, resolution, and codec information to detect deepfake patterns:  
```bash
python scripts/extract_metadata.py
```

---

## 📜 License  
This project is licensed under the **MIT License**.  

---

Contributions welcome! 😊 🚀

