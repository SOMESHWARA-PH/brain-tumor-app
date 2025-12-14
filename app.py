import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image to detect tumor.")

@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
