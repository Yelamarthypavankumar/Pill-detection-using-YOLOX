import cv2
import numpy as np
import onnxruntime
import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "yolox"))

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, vis

st.set_page_config(layout="wide")

st.title("PILL DETECTION")

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    mode = st.radio("Select Mode", ["Live Camera", "Image Upload"])
    days = st.number_input("Enter the number of days needed:", min_value=1, step=1)
    times = st.number_input("Enter the number of times needed:", min_value=1, step=1)
    tablets_needed = days * times

# ONNX model setup
session = onnxruntime.InferenceSession("yolox_s_pill.onnx")
input_shape = (640, 640)
custom_classes = ["medications", "capsule", "pill"]

def process_image(frame, tablets_needed):
    img, ratio = preprocess(frame, input_shape)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    
    try:
        output = session.run(None, ort_inputs)
    except Exception as e:
        st.error(f"Inference failed: {e}")
        return frame, "", 0
    
    predictions = demo_postprocess(output[0], input_shape)[0]
    if predictions is None:
        return frame, "<h3 style='color:red;'>No objects detected.</h3>", 0

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.4, score_thr=0.2)
    processed_frame = frame.copy()
    message = ""

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        processed_frame = vis(processed_frame, final_boxes, final_scores, final_cls_inds,
                              conf=0.3, class_names=custom_classes)
        detected_count = len(final_cls_inds)

        if detected_count == tablets_needed:
            message = f"<h2 style='color:green;'>{tablets_needed} pills detected </h2>"
        elif detected_count < tablets_needed:
            message = f"<h2 style='color:red;'>You need to add {tablets_needed - detected_count} pills.</h2>"
        else:
            message = f"<h2 style='color:red;'>You need to remove {detected_count - tablets_needed} pills.</h2>"
    else:
        message = "<h3 style='color:red;'>No pills detected.</h3>"

    return processed_frame, message, detected_count

# Live camera setup
if "cap" not in st.session_state:
    st.session_state.cap = None
    st.session_state.stop = True
    st.session_state.message = ""

def start_stream():
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        st.session_state.cap = cv2.VideoCapture(0)  # Change index if needed
    st.session_state.stop = False

def stop_stream():
    st.session_state.stop = True
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    st.session_state.message = ""

if mode == "Live Camera":
    st.button("Start Stream", on_click=start_stream)
    st.button("Stop Stream", on_click=stop_stream)

    with col2:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()

    with col3:
        st.subheader("Processed Output")
        output_placeholder = st.empty()
        message_placeholder = st.empty()

    if not st.session_state.stop:
        cap = st.session_state.cap
        if cap is None or not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
            else:
                processed_frame, message, count = process_image(frame, tablets_needed)
                frame_placeholder.image(frame, channels="BGR")
                output_placeholder.image(processed_frame, channels="BGR")
                message_placeholder.markdown(message, unsafe_allow_html=True)

elif mode == "Image Upload":
    with col2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    with col3:
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, channels="BGR", caption="Uploaded Image")
            
            processed_frame, message, count = process_image(image, tablets_needed)
            st.image(processed_frame, channels="BGR", caption="Processed Output")
            st.markdown(message, unsafe_allow_html=True)

