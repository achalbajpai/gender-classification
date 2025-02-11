import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set page config
st.set_page_config(page_title="Gender Classification", page_icon="👤", layout="wide")


# Model class
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))
    model.load_state_dict(
        torch.load("best_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()
    return model


# Image transformation
def transform_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


# Prediction function
def predict(model, image):
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()
        return probability


# Main app
def main():
    st.title("Gender Classification Model")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Home", "Model Details", "Test Model", "Performance Metrics"]
    )

    if page == "Home":
        st.header("Welcome to Gender Classification App")
        st.write(
            """
        This application uses a deep learning model (ResNet18) to classify gender from facial images.
        The model has been trained on a large dataset and achieves 98.22% accuracy in gender prediction.
        
        ### Features:
        - Upload and test your own images
        - View detailed model architecture and training process
        - Examine model performance metrics
        - Real-time predictions with confidence scores
        """
        )

        # Display sample images if available
        if os.path.exists("Dataset/Test"):
            st.subheader("Sample Images from Test Set")
            test_path = Path("Dataset/Test")
            sample_images = list(test_path.glob("*/*"))[:4]
            cols = st.columns(4)
            for col, img_path in zip(cols, sample_images):
                with col:
                    st.image(str(img_path), caption=img_path.parent.name)

    elif page == "Model Details":
        st.header("Model Architecture and Training Details")

        # Load and display technical details
        with open("technical_details.md", "r") as file:
            st.markdown(file.read())

    elif page == "Test Model":
        st.header("Test the Model")

        # Load model
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                # Display original image
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)

                # Make prediction
                transformed_image = transform_image(image)
                probability = predict(model, transformed_image)

                # Display results
                st.subheader("Prediction Results:")
                col1, col2 = st.columns(2)

                with col1:
                    gender = "Male" if probability > 0.5 else "Female"
                    confidence = probability if probability > 0.5 else 1 - probability
                    st.write(f"Predicted Gender: **{gender}**")
                    st.write(f"Confidence: **{confidence*100:.2f}%**")

                with col2:
                    # Create confidence visualization
                    fig, ax = plt.subplots(figsize=(8, 1))
                    ax.barh(["Confidence"], [confidence * 100], color="blue")
                    ax.set_xlim(0, 100)
                    plt.tight_layout()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif page == "Performance Metrics":
        st.header("Model Performance Metrics")

        # Display test results
        st.subheader("Test Set Results")
        st.code(
            """
Test Accuracy: 98.22%

Classification Report:
              precision    recall  f1-score   support

      Female       0.98      0.99      0.98      2851
        Male       0.98      0.98      0.98      2149

    accuracy                           0.98      5000
   macro avg       0.98      0.98      0.98      5000
weighted avg       0.98      0.98      0.98      5000
        """
        )

        # Display training metrics
        if os.path.exists("training_metrics.png"):
            st.subheader("Training Progress")
            st.image("training_metrics.png", caption="Training and Validation Metrics")

        # Display ROC curve
        if os.path.exists("roc_curve.png"):
            st.subheader("ROC Curve")
            st.image("roc_curve.png", caption="Receiver Operating Characteristic Curve")

        # Display confusion matrix
        if os.path.exists("confusion_matrix.png"):
            st.subheader("Confusion Matrix")
            st.image("confusion_matrix.png", caption="Test Set Confusion Matrix")


if __name__ == "__main__":
    main()
