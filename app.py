import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Brain Tumor Detection")
st.title("🧠 Brain Tumor Detection")

# Load ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

session = load_model()
input_name = session.get_inputs()[0].name

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

def preprocess(image):
    image = np.array(image)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    output = session.run(None, {input_name: img})
    pred = np.argmax(output[0])
    confidence = np.max(output[0]) * 100

    if pred == 1:
        st.error(f"🧠 Tumor Detected ({confidence:.2f}%)")
    else:
        st.success(f"✅ No Tumor Detected ({confidence:.2f}%)")
