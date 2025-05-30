# 💊 Pill Detection App using YOLOX, ONNXRuntime, OpenCV, and Streamlit

This is a real-time pill detection application built with **YOLOX**, **ONNXRuntime**, **OpenCV**, and **Streamlit**. It enables live and offline detection of pills and provides automated pill counting and validation based on dosage inputs.

---

## ✅ Key Features

- 📷 Live webcam stream for real-time pill detection  
- 🖼️ Image upload for offline detection  
- 🔢 Automatic pill counting and dosage validation  
- 💡 Real-time feedback on dosage compliance  

---

## 🖼️ App UI Features

- Upload or stream images of pills
- Specify the required number of pills per day and dosage frequency
- Detects pills, capsules, and tablets using a custom-trained YOLOX ONNX model
- Provides instant guidance: whether to **add** or **remove** pills based on your daily dosage

---

## ⚙️ How to Run the Inference

### 🧪 Step-by-Step Setup

1. **Create a Conda environment**

   ```bash
   conda create -n pill_detection python=3.10
   conda activate pill_detection

2.Clone the YOLOX repository
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git yolox
3.Install dependencies
```bash
pip install -r requirements.txt
4.Run the Streamlit app
```bash
streamlit run inference.py
```
🖼️ Sample Test Images
You can find example images to test the app in the Images/ folder.
![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/8efa46d7-22cd-4388-b3b9-c9063f324db3)


