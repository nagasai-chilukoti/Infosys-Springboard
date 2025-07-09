import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from torchvision import models

# ----------------------------
# Model Definition (must match training)
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    pil_image = Image.fromarray(edges_rgb)
    return transform(pil_image).unsqueeze(0)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Iris Tumor Detection", layout="centered")
st.title("🔬 Iris Tumor Detection App (ResNet + PyTorch)")

st.write(
    "Upload an eye image, and our AI model will predict whether a tumor is present. "
    "This app uses **PyTorch ResNet18** for deep learning-based classification."
)

uploaded_file = st.file_uploader("📤 Upload an Eye Image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="✅ Uploaded Image", use_column_width=True)
    with st.spinner("Analyzing..."):
        input_tensor = preprocess_image(uploaded_file)

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

        # Optional: display probability breakdown
        st.write("### 📊 Probability Breakdown")
        st.progress(tumor_prob)
        st.write(f"Tumor Probability: **{tumor_prob:.2%}**")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This model uses a **PyTorch ResNet18** architecture trained on processed iris images "
    "to classify the presence of a tumor.\n\nDeveloped by [nagasai chilukoti]."
)

st.sidebar.markdown("📧 Contact: **nagasaichilukoti71@gmail.com**")