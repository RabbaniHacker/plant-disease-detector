import torch, json
import numpy as np
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")
st.markdown("""
<style>
    h1 { color: #1b5e20; }
    .stProgress > div > div > div > div { background-color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = json.load(open("class_names.json"))

@st.cache_resource
def load_model():
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
    m.to(device).eval()
    return m

model = load_model()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def pretty(name):
    return name.replace("___", " - ").replace("_", " ")

# Simple treatment tips (extend as you like)
TIPS = {
    "healthy": "Plant looks healthy. Maintain regular watering and monitoring.",
    "blight": "Remove affected leaves, avoid overhead watering, apply a copper-based fungicide.",
    "rust": "Remove infected leaves, improve air circulation, apply fungicide if severe.",
    "spot": "Prune affected areas, avoid wetting foliage, use appropriate fungicide.",
    "mold": "Increase ventilation, reduce humidity, apply fungicide.",
    "virus": "No cure; remove and destroy infected plants to prevent spread. Control insect vectors.",
    "mites": "Spray with water or insecticidal soap; introduce predatory mites.",
    "scab": "Remove fallen leaves, apply fungicide in early season.",
}
def get_tip(name):
    low = name.lower()
    for k, v in TIPS.items():
        if k in low:
            return v
    return "Consult a local agricultural expert for treatment guidance."

# Grad-CAM on the last conv layer of ResNet18 (layer4)
def grad_cam(input_tensor, class_idx):
    feats, grads = {}, {}
    def fwd(m, i, o): feats["v"] = o.detach()
    def bwd(m, gi, go): grads["v"] = go[0].detach()
    h1 = model.layer4.register_forward_hook(fwd)
    h2 = model.layer4.register_full_backward_hook(bwd)
    out = model(input_tensor)
    model.zero_grad()
    out[0, class_idx].backward()
    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = (w * feats["v"]).sum(1).squeeze().relu().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay(img, cam):
    import matplotlib.cm as cm
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size)
    heat = (cm.jet(np.array(cam_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.5 * np.array(img) + 0.5 * heat).astype(np.uint8)
    return Image.fromarray(blended)

st.title("🌿 Plant Disease Detector")
st.caption("Detect plant diseases from a leaf image. ResNet18 fine-tuned to 99.1% accuracy on 38 classes, with Grad-CAM explainability.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0]
        top_probs, top_idx = probs.topk(3)
    best_idx = top_idx[0].item()
    best = classes[best_idx]

    cam = grad_cam(x.clone().requires_grad_(True), best_idx)
    cam_img = overlay(img.resize((224, 224)), cam)

    c1, c2 = st.columns(2)
    c1.image(img, caption="Uploaded image", use_container_width=True)
    c2.image(cam_img, caption="Grad-CAM: regions the model focused on", use_container_width=True)

    if "healthy" in best.lower():
        st.success(f"✅ {pretty(best)}")
    else:
        st.error(f"⚠️ {pretty(best)}")
    st.metric("Confidence", f"{top_probs[0].item()*100:.1f}%")

    st.subheader("Top 3 predictions")
    for p, idx in zip(top_probs, top_idx):
        st.write(pretty(classes[idx.item()]))
        st.progress(float(p))

    st.subheader("💡 Treatment guidance")
    st.info(get_tip(best))
else:
    st.info("👆 Upload a leaf image to get started.")
