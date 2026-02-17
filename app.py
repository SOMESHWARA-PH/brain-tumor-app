import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Detection")

st.title("🧠 Brain Tumor Detection")

# ---------------- DEMO MODEL ----------------
@st.cache_resource
def load_model():
    return "demo_model"   # demo mode

model = load_model()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    image = np.repeat(image, 3, axis=-1)

    return image

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=350)

    _ = preprocess_image(image)  # preprocessing demo

    # ---------------- DEMO PREDICTION ----------------
    confidence = 0.78  # demo value

    st.error(f"❌ Tumor Detected ({confidence*100:.2f}%)")

    st.markdown("---")
    st.subheader("📝 Precautions & Health Tips")

    st.markdown("""
    **⚠️ Tumor Detected – Suggested Precautions:**
    - Consult a **neurologist or neurosurgeon**
    - Avoid stress and take **proper rest**
    - Follow **MRI / CT scan** advice from doctor
    - Do **not self-medicate**
    - Maintain a **healthy diet**
    """)

    st.info("⚠️ Result shown in **Demo Mode** for presentation")
