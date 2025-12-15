import streamlit as st
from PIL import Image

# Page config
st.set_page_config(page_title="Brain Tumor Detection")

# Title
st.title("🧠 Brain Tumor Detection")

# Status message
st.success("✅ Streamlit app is running correctly")
st.write("If you see this message, deployment is successful.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

# Show uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=700)
    st.info("Model integration will be added next")


