import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Define transformations (same as the validation transform)
val_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load EfficientNet-B7 from torchvision and adjust for 2 classes
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.efficientnet_b7(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)

    checkpoint_path = os.path.join('..', 'checkpoints', 'best_checkpoint.pth.tar')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)
    model.eval()
    return model, device


def predict_image_with_probability(model, device, image):
    image = val_transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probability = probabilities[0, predicted_class].item()

    return predicted_class, predicted_probability


def app():
    st.title("Rip Current Classifier")
    st.write("Upload a beach image, and the app will predict whether the water is safe to enter based on the rip current.")
    st.write("Model classes: **Yes** (dangerous rip current) or **No** (safe).")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    class_names = ['no', 'yes']

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        model, device = load_model()
        predicted_class, predicted_probability = predict_image_with_probability(model, device, image)

        st.write(f"**Prediction:** {class_names[predicted_class].upper()}")
        st.write(f"**Confidence:** {predicted_probability * 100:.2f}%")
