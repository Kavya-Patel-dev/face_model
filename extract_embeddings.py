import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms
import numpy as np
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(image_size=160, margin=0).to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

DATASET_DIR = "dataset/"

embeddings = []
labels = []


for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    
    # Augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        
        try:
            original_img = Image.open(img_path).convert("RGB")
        except:
            continue
            
        # Generate original + 5 augmented versions
        images_to_process = [original_img]
        for _ in range(5):
            images_to_process.append(transform(original_img))
            
        for img in images_to_process:
            face = mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
                embeddings.append(emb.flatten())
                labels.append(person)

embeddings = np.array(embeddings)
labels = np.array(labels)

joblib.dump((embeddings, labels), "embeddings.pkl")
print("Embeddings saved!")
