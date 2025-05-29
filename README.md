# Pill-detection-using-YOLOX
This is a real-time pill detection application built with YOLOX, ONNXRuntime, OpenCV, and Streamlit. It supports:

✅ Live webcam stream for detection

✅ Image upload for offline detection

✅ Automatic pill counting and validation based on dosage input

🖼️ App UI Features
Upload or stream pills to count

Specify number of pills required per day and how many times per day

Detects pills (pill, capsule, medications) using a YOLOX ONNX model

Tells if pills are missing or in excess
To run the inference:
create the env using Conda
conda create env pill_detection python=3.10
then
git clone https://github.com/Megvii-BaseDetection/YOLOX.git yolox
then 
pip install -r requirements.txt
then
streamlit run inference.py
I provide some examples to text in the Images folder 
