import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection")
st.success("App deployed successfully without TensorFlow")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.info("Model loading will be added next")

