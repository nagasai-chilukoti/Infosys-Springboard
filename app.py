import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from torchvision import models

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #F5F7FA;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .uploadedImage {
        border: 2px solid #4CAF50;
        padding: 5px;
        border-radius: 10px;
        display: block;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Model Definition
# ----------------------------
class ResNetIrisTumor(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(ResNetIrisTumor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = ResNetIrisTumor()
    model.load_state_dict(torch.load("iris_tumor_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(edges_rgb)
    return transform(pil_image).unsqueeze(0), image

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Iris Tumor Detection", layout="centered")
st.title("🔬 Iris Tumor Detection App")
st.write("""
Upload an eye image, and our AI model will predict whether a tumor is present. 
This tool is built using **ResNet18 (PyTorch)** for accurate predictions.
""")

# 📂 File Uploader
uploaded_file = st.file_uploader("📤 Upload an Eye Image...", type=["jpg", "png", "jpeg"])

# 🌙 Dark Mode Toggle
if st.sidebar.checkbox("🌙 Dark Mode"):
    st.markdown("""
        <style>
        .main {
            background-color: #222;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# 📷 Image Preview & Prediction
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    input_tensor, raw_image = preprocess_image(uploaded_file)

    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(raw_image, caption="✅ Image Preview", use_container_width=True)

    with col2:
        st.subheader("🧪 Prediction Results")
        with torch.no_grad():
            output = model(input_tensor)
            _, prediction = torch.max(output, 1)
            probs = torch.softmax(output, dim=1)[0]
            tumor_prob = probs[1].item()

        label = "Tumor Detected" if prediction.item() == 1 else "No Tumor"
        confidence = tumor_prob if prediction.item() == 1 else 1 - tumor_prob

        if prediction.item() == 1:
            st.error(f"🚨 {label} ({confidence:.2%} confidence)")
        else:
            st.success(f"✅ {label} ({confidence:.2%} confidence)")

        st.write("### 📊 Probability Breakdown")
        st.progress(tumor_prob)
        st.write(f"Tumor Probability: **{tumor_prob:.2%}**")

# Sidebar
st.sidebar.subheader("ℹ️ About This Model")
st.sidebar.write("""
This model uses **ResNet18 (CNN)** trained on processed iris images 
to detect tumors with high accuracy.
Developed by **nagasai chilukoti** for healthcare research.
""")

st.sidebar.subheader("📩 Contact Us")
st.sidebar.write("📧 Email: **nagasaichilukoti71@gmail.com**")
