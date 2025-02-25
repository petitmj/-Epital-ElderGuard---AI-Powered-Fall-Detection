# 📌 Epital ElderGuard - AI-Powered Fall Detection

**Real-time fall detection and activity monitoring for elderly care using AI models deployed on Snapdragon-powered devices.**  

---

## 📖 Table of Contents
- [Introduction](#introduction)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training & Conversion](#model-training--conversion)  
- [On-Device Deployment](#on-device-deployment)  
- [Streamlit App for Insights](#streamlit-app-for-insights)  
- [Benchmark Results](#benchmark-results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🔥 Introduction
Epital ElderGuard is an **AI-powered fall detection system** designed for **elderly safety**. It combines **machine learning models (Autoencoder, Isolation Forest)** with **Qualcomm AI Hub** for **on-device inference** on **Snapdragon-powered devices** like the **Samsung Galaxy S23/S24**. The system provides **actionable insights** for caregivers, ensuring **fast response times and enhanced elderly care**.  

---

## 🚀 Features
✅ **Real-time fall detection** with on-device inference  
✅ **Hybrid AI models** (Autoencoder + Isolation Forest)  
✅ **Qualcomm AI Hub deployment** for optimized performance  
✅ **TFLite & ONNX model conversions**  
✅ **Streamlit dashboard for caregiver insights**  
✅ **Minimal latency (~4.3ms)** on Snapdragon devices  
✅ **Benchmark testing & quantization for mobile efficiency**  

---

## 🛠 Tech Stack
- **Machine Learning**: TensorFlow, scikit-learn, ONNX  
- **Deployment**: Qualcomm AI Hub, TFLite, ONNX Runtime  
- **Backend**: FastAPI  
- **Frontend**: Streamlit  
- **Hardware**: Snapdragon 8 Gen 2 (Samsung Galaxy S23/S24)  

---

## 🛠 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/petitmj/epital-elderguard.git  
cd epital-elderguard  
pip install -r requirements.txt  
```
Ensure **Qualcomm AI Hub SDK** is set up:  
```bash
export PATH=$PATH:/path/to/qai_hub/bin  
```

---

## 🏗 Usage
### 1️⃣ Train & Convert Models
Train the models and convert them to **TFLite & ONNX**:  
```bash
python models/model_conversion.py  
```
  
### 2️⃣ Deploy on Qualcomm AI Hub
Upload and deploy the **optimized ONNX model**:  
```bash
python scripts/deploy_aihub.py  
```
  
### 3️⃣ Run On-Device Inference
Perform real-time inference on a **Snapdragon device**:  
```bash
python scripts/run_inference.py  
```
  
### 4️⃣ Streamlit App for Insights(FUTURE VERSION)
Launch the **caregiver dashboard**:  
```bash
streamlit run app.py  
```
  
---

## 🎯 Model Training & Conversion
1. **Train Hybrid Models** (Autoencoder + Isolation Forest)  
2. **Convert Autoencoder to TFLite**  
3. **Convert Isolation Forest to ONNX**  
4. **Optimize ONNX model for Snapdragon devices**  

---

## 📲 On-Device Deployment
- **Device:** Samsung Galaxy S23/S24 (Snapdragon 8 Gen 2)  
- **Inference Time:** **~4.3ms**  
- **Memory Usage:** **0 - 11MB**  
- **Optimization:** Quantization applied for mobile  

---

## 📊 Benchmark Results
| Metric               | Value  |  
|----------------------|--------|  
| **Inference Time**   | 4.3ms  |  
| **Memory Usage**     | 0-11MB |  
| **Model Size (TFLite)** | 1.2MB |  
| **Model Size (ONNX Optimized)** | 850KB |  

---

## 👨‍💻 Contributing
We welcome contributions! 🚀  
1. Fork the repo  
2. Create a feature branch (`feature-xyz`)  
3. Commit changes (`git commit -m "Added new feature"`)  
4. Push to branch (`git push origin feature-xyz`)  
5. Submit a PR  

---

## 📜 License
**MIT License** - Free to use and modify.  

---

🚀 **Let's redefine elderly safety with AI!** 🔥  


