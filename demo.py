import faiss
import numpy as np
import torch
import os
import json
from PIL import Image
from googletrans import Translator
import matplotlib.pyplot as plt
import glob
import math
from support_models.CLIP.clip import clip

# Loadthe pre-trained CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# Load the feature vectors
faiss_index_dir = "/home/students/ttnhan-cse23/hcmai/dict/faiss_normal_ViT.bin"
index = faiss.read_index(faiss_index_dir)

# index in json file
id2filename = {}
with open("/home/students/ttnhan-cse23/hcmai/dict/image_path.json", "r") as f:
    id2filename = json.load(f)

# image folder
image_folder = "/home/students/ttnhan-cse23/hcmai/dataset/extracted_frames"

# ENCODE TEXT AND IMAGE
def encode_text(text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def encode_images(images):
    with torch.no_grad():
        image_features = torch.cat([model.encode_image(image) for image in images])
    return image_features

def display_similar_images(image_paths,index,text, num_images=5):
    print(f"Text Query: {text}")
    translator = Translator()
    detected = translator.detect(text)
    if detected.lang == 'vi':
      translated = translator.translate(text, src='vi', dest='en')
      text=translated.text
      print(f"Text Query Translated: {text}")
    text_features = encode_text(text)
    
    k = max(1, min(len(image_paths), num_images))
    scores, top_k = index.search(text_features.cpu().numpy().astype("float32"), k)

    scores = scores[0]
    top_k = top_k[0]
    
    image_filenames = [image_folder + '/' + id2filename[str(image_id)] for image_id in top_k]
    
    results  = [f"{image_filename.split('/')[1]}_{image_filename.split('/')[2]}, {image_filename.split('/')[3][:-4]}" for image_filename in image_filenames]
    image_results = zip(image_filenames, results)

    print("Top matches:")

    cols = min(5, num_images)
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = np.array(axes).flatten() if rows > 1 else np.array([axes])
    
    for i in range(len(image_filenames)):
        img = Image.open(image_filenames[i])
        axes[i].imshow(img)
        axes[i].set_title(f"Score: {scores[i]:.2f}", fontsize=10)
        axes[i].axis("off")
        
    for j in range(len(image_filenames), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show(block=True)

if __name__ ==  "__main__":
    images = []
    image_paths = []
    
    for folder in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder)
        for video in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video)
            for filename in os.listdir(video_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
                    image_path = os.path.join(video_path, filename)
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    images.append(image)
                    image_paths.append(image_path)
    
    text_query = "A zookeeper or animal caretaker wearing a black uniform is kneeling beside a king penguin outdoors. The penguin is standing upright on a paved surface, looking ahead"
    display_similar_images(image_paths, index, text_query, num_images=100)
