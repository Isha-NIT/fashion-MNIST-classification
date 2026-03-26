import streamlit as st
import requests
import os
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "https://ishaks2005-fashion-mnist.hf.space")

st.set_page_config(page_title="Fashion MNIST Classifier", layout="centered")
st.title("Fashion-MNIST Image Classifier")
st.write("Upload an image and get a prediction")

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)   # Converts uploaded file into a PIL Image object
    st.image(image, use_column_width=True)

    if st.button("Predict"):
        try:
            with st.spinner("Predicting..."):
                uploaded_file.seek(0)
                files = {"file": ("image.png", uploaded_file.getvalue(), "image/png")}
    
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files=files,
                    timeout=15
                )
                
                st.write("Status:", response.status_code)
                
                if response.status_code == 200:   # 200 = Success
                    result = response.json()
                    st.success(f"Prediction: **{result['prediction']}**")
                    st.info(f"Confidence: **{result['confidence']}%**")
                else:
                    st.error("Error calling FastAPI backend")

        except Esception as e:
            st.error(f"Error: {e}")


