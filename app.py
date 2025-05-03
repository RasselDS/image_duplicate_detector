import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import gradio as gr
import os
import shutil
import tempfile
from pathlib import Path
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from fastai.vision.all import *
from uuid import uuid4


# Create image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load pretrained model
model = resnet18(pretrained=True)
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

# Remove the classification head
embedding_model = torch.nn.Sequential(*list(model.children())[:-1])

# Compute embedding
def get_embedding(img: Image.Image):
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.cuda() if torch.cuda.is_available() else img_tensor
    with torch.no_grad():
        emb = embedding_model(img_tensor).squeeze().flatten()
    return emb.cpu()

# Merge images side by side for display
def merge_images_side_by_side(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB").resize((256, 256))
    img2 = Image.open(img2_path).convert("RGB").resize((256, 256))
    merged = Image.new('RGB', (512, 256))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (256, 0))
    return merged

# Main function
def find_duplicates(files, threshold=0.9):
    temp_dir = Path(tempfile.mkdtemp())
    image_paths = []

    for f in files:
        ext = Path(f.name).suffix
        temp_path = temp_dir / f"{uuid4()}{ext}"
        shutil.copy(f.name, temp_path)
        image_paths.append(temp_path)

    embeddings = {}
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        embeddings[img_path] = get_embedding(img)

    duplicates = []
    for (img1, emb1), (img2, emb2) in combinations(embeddings.items(), 2):
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))[0][0]
        if sim > threshold:
            merged_img = merge_images_side_by_side(img1, img2)
            merged_path = temp_dir / f"merged_{uuid4().hex}.jpg"
            merged_img.save(merged_path)
            duplicates.append((str(merged_path), f"Similarity: {sim:.2f}"))

    if not duplicates:
        return "No duplicates found!", []
    return f"Found {len(duplicates)} likely duplicate pairs", duplicates

# Gradio interface
inputs = [
    gr.File(file_types=["image"], file_count="multiple", label="Upload Images"),
    gr.Slider(0.5, 1.0, step=0.01, value=0.7, label="Similarity Threshold")
]

outputs = [
    gr.Textbox(label="Result"),
    gr.Gallery(label="Duplicate Pairs", show_label=True, columns=2)
]

app = gr.Interface(fn=find_duplicates, inputs=inputs, outputs=outputs, title="Image Duplicate Detector")
app.launch(share=True)
