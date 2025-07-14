import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# === CONFIG ===
train_dir = 'dataset_split/train'
batch_size = 32  # Increased for speed
epochs = 5       # Reduced for faster training
lr = 0.001
embedding_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset Loader ===
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        label_id = 0

        for person_id in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_id)
            if not os.path.isdir(person_path):
                continue
            for name_subdir in os.listdir(person_path):
                name_path = os.path.join(person_path, name_subdir)
                if not os.path.isdir(name_path):
                    continue
                label = f"{person_id}_{name_subdir}"
                if label not in self.label_map:
                    self.label_map[label] = label_id
                    label_id += 1
                for quality_subdir in os.listdir(name_path):
                    quality_path = os.path.join(name_path, quality_subdir)
                    if not os.path.isdir(quality_path):
                        continue
                    for img_file in os.listdir(quality_path):
                        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                            full_path = os.path.join(quality_path, img_file)
                            self.image_paths.append(full_path)
                            self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, FileNotFoundError):
            print(f"⚠️ Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))
        if self.transform:
            image = self.transform(image)
        return image, label

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Load Data ===
train_dataset = FaceDataset(train_dir, transform)
print(f"🧾 Total training images found: {len(train_dataset)}")
if len(train_dataset) == 0:
    raise ValueError("🚫 No images found in training directory. Check path or folder content.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Model ===
class FaceNet(nn.Module):
    def __init__(self, embedding_size=128, num_classes=100):
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

model = FaceNet(embedding_size=embedding_size, num_classes=len(train_dataset.label_map)).to(device)

# === Training Setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("🚀 Starting Fine-Tuning...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"📘 Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "fine_tuned_model.pth")
print("✅ Fine-tuned model saved as 'fine_tuned_model.pth'")


