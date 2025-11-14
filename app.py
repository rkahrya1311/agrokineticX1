import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plantvillage_limited_model.h5")

model = load_model()

st.title("🌱 PlantVillage Disease Detector")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess image (adjust size to your model’s input)
    img = image.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    prediction = model.predict(arr)

    # Example: if your model outputs probabilities for classes
    classes = ["Healthy", "Early Blight", "Late Blight"]  # replace with your actual labels
    predicted_class = classes[np.argmax(prediction)]

    st.subheader("Prediction")
    st.write(f"🟢 {predicted_class}")

    # Advisory text (Tamil + English example)
    if predicted_class == "Healthy":
        st.success("✅ Crop is healthy.\n\nதமிழ்: பயிர் ஆரோக்கியமாக உள்ளது.")
    elif predicted_class == "Early Blight":
        st.warning("⚠️ Early Blight detected.\n\nதமிழ்: ஆரம்ப பிளைட் நோய் கண்டறியப்பட்டது. உடனடி சிகிச்சை தேவை.")
    elif predicted_class == "Late Blight":
        st.error("❌ Late Blight detected.\n\nதமிழ்: கடைசி பிளைட் நோய் கண்டறியப்பட்டது. விரைவில் நடவடிக்கை எடுக்கவும்.")
