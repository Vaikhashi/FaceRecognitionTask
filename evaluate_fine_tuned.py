import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# === CONFIG ===
val_dir = 'dataset_split/val'
model_path = 'fine_tuned_model.pth'
batch_size = 16
embedding_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Loader ===
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.label_map = {}
        label_id = 0

        for person_name in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            for name_folder in os.listdir(person_path):
                name_path = os.path.join(person_path, name_folder)
                if not os.path.isdir(name_path):
                    continue
                if name_folder not in self.label_map:
                    self.label_map[name_folder] = label_id
                    label_id += 1
                for quality_folder in os.listdir(name_path):
                    quality_path = os.path.join(name_path, quality_folder)
                    if not os.path.isdir(quality_path):
                        continue
                    for img_file in os.listdir(quality_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.image_paths.append(os.path.join(quality_path, img_file))
                            self.labels.append(self.label_map[name_folder])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Skipping unreadable: {img_path}")
            return None
        if self.transform:
            image = self.transform(image)
        return image, label

# === Custom Collate Function ===
def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

# === Model Definition ===
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128, num_classes=5):
        super(FaceNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(512, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        emb = self.embedding(x)
        x = self.classifier(emb)
        return x, emb

# === Load Validation Dataset ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_dataset = FaceDataset(val_dir, transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
print(f"📦 Total validation images found: {len(val_dataset)}")

# === Load Model ===
model = FaceNet(num_classes=len(val_dataset.label_map))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("🚀 Evaluating Fine-Tuned Model...")

embeddings = []
labels = []

with torch.no_grad():
    for images, lbls in val_loader:
        if images.nelement() == 0:  # empty batch due to all skipped
            continue
        images = images.to(device)
        _, embs = model(images)
        embeddings.append(embs.cpu().numpy())
        labels.extend(lbls)

embeddings = np.vstack(embeddings)
labels = np.array(labels)

# === Pairwise Evaluation ===
similarities = []
true_labels = []

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        label = 1 if labels[i] == labels[j] else 0
        similarities.append(sim)
        true_labels.append(label)

similarities = np.array(similarities)
true_labels = np.array(true_labels)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(true_labels, similarities)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Post Fine-Tuning")
plt.legend()
plt.savefig("roc_finetuned.png")
plt.close()

# === Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(true_labels, similarities)
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Post Fine-Tuning")
plt.savefig("precision_recall_finetuned.png")
plt.close()

# === Similarity Histogram ===
pos_sims = similarities[true_labels == 1]
neg_sims = similarities[true_labels == 0]

plt.figure()
plt.hist(pos_sims, bins=50, alpha=0.6, label='Same')
plt.hist(neg_sims, bins=50, alpha=0.6, label='Different')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Similarity Distribution - Post Fine-Tuning")
plt.legend()
plt.savefig("similarity_dist_finetuned.png")
plt.close()

print("✅ Evaluation Complete. Plots saved:")
print("- roc_finetuned.png")
print("- precision_recall_finetuned.png")
print("- similarity_dist_finetuned.png")


