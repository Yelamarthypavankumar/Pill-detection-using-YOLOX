💊 Pill Detection App using YOLOX, ONNXRuntime, OpenCV, and Streamlit
This is a real-time pill detection application built with YOLOX, ONNXRuntime, OpenCV, and Streamlit.

✅ Key Features
Live webcam stream for detection

Image upload for offline detection

Automatic pill counting and validation based on dosage input

🖼️ App UI Features
Upload or stream images of pills

Specify number of pills required per day and how many times per day

Detects pills (pill, capsule, medications) using a custom YOLOX ONNX model

Provides real-time feedback: whether to add or remove pills based on the dosage calculation

⚙️ How to Run the Inference
🧪 Step-by-Step Setup
Create a Conda environment:


conda create -n pill_detection python=3.10
conda activate pill_detection
Clone the YOLOX repository:


git clone https://github.com/Megvii-BaseDetection/YOLOX.git yolox
Install dependencies:


pip install -r requirements.txt
Run the Streamlit app:


streamlit run inference.py
🖼️ Sample Test Images
You can find example images to test the app in the Images/ folder.

🛠️ Requirements (requirements.txt)
Here is a sample requirements.txt for your reference:


streamlit
opencv-python
onnxruntime
numpy
Add other packages you use (e.g., matplotlib, Pillow) as needed.
