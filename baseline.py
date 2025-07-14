import os
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# === Setup ===
dataset_path = "images"
output_path = "embeddings"
log_path = "processed_logs"
os.makedirs(output_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def try_face_analysis(img, app):
    faces = app.get(img)
    if len(faces) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        faces = app.get(gray)
    return faces

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(320, 320))

def adjust_gamma(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

print("🔍 Starting face embedding extraction...")

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)
    if not os.path.isdir(person_path): continue

    for name_folder in os.listdir(person_path):
        name_path = os.path.join(person_path, name_folder)
        if not os.path.isdir(name_path): continue

        for quality_folder in os.listdir(name_path):
            quality_path = os.path.join(name_path, quality_folder)
            if not os.path.isdir(quality_path): continue

            image_files = [f for f in os.listdir(quality_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not image_files: continue

            embeddings = []
            processed_files = []
            save_name = f"{name_folder}_{quality_folder}.npy"
            log_file = os.path.join(log_path, f"{name_folder}_{quality_folder}_used.txt")

            for img_file in tqdm(image_files, desc=f"Processing {name_folder}/{quality_folder}"):
                img_path = os.path.join(quality_path, img_file)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.resize(img, (320, 320))
                img = adjust_gamma(img, gamma=1.2)

                faces = try_face_analysis(img, app)
                if len(faces) == 0:
                    continue

                emb = faces[0].embedding
                embeddings.append(emb)
                processed_files.append(img_file)

            if embeddings:
                np.save(os.path.join(output_path, save_name), np.array(embeddings))
                with open(log_file, 'w') as f:
                    for name in processed_files:
                        f.write(name + '\n')
                print(f"✅ Saved: {save_name} | {len(embeddings)} embeddings")
            else:
                print(f"❌ No faces found for: {name_folder}/{quality_folder}")

print("🎉 Done! All usable embeddings saved.")
