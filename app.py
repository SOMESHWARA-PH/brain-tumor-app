import streamlit as st
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection")
st.success("Streamlit app is working correctly")

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.info("Model integration will be added later")
