from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.image_classifier.infer import predict_image


# Configure Streamlit page title and layout
st.set_page_config(page_title="Flower Species Classifier", layout="centered")


def format_confidence(score: float) -> str:
    # Convert probability to human-readable percentage
    return f"{score * 100:.2f}%"


# Page header and short description
st.title("Flower Species Classifier")
st.write("Upload a flower image to see the predicted species and confidence.")

# Text input for checkpoint path and file uploader for images
checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/flower_classifier.pt")
uploaded_file = st.file_uploader("Choose a flower image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Load checkpoint path and run prediction if checkpoint exists
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists():
        prediction = predict_image(image, checkpoint)
        st.subheader("Prediction")
        st.write(f"Species: **{prediction.label}**")
        st.write(f"Confidence: **{format_confidence(prediction.confidence)}**")

        # Show top-k predictions in an expander
        with st.expander("Top predictions"):
            for label, score in prediction.top_k:
                st.write(f"{label}: {format_confidence(score)}")
    else:
        # Error if checkpoint not found
        st.error("Checkpoint not found. Train the model first or enter an existing checkpoint.")
