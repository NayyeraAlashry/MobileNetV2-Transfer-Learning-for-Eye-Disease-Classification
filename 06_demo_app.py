import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# 1. Recreate the improved model architecture
def load_trained_model(weights_path="mobilenet_v2_finetuned.pth"):
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 4)
    )
    # Load trained weights on CPU
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_trained_model()

# 2. Define prediction transform matching training pipeline
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# 3. Prediction function
def predict_eye_disease(image):
    if image is None:
        return {}
    
    # Ensure RGB format
    image = image.convert("RGB")
    tensor = inference_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

# 4. Build Gradio Web Interface
demo = gr.Interface(
    fn=predict_eye_disease,
    inputs=gr.Image(type="pil", label="Upload Retinal Fundus Image"),
    outputs=gr.Label(num_top_classes=4, label="Diagnosis Confidence"),
    title="Eye Disease Diagnostic System",
    description="Deep Learning diagnostic system powered by Fine-Tuned MobileNetV2 for detecting Cataract, Diabetic Retinopathy, Glaucoma, and Normal retina."
)

if __name__ == "__main__":
    demo.launch()
