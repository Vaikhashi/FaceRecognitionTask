import os
import random
import shutil

source_root = "images"
output_root = "dataset_split"
train_ratio = 0.8

for person_id in os.listdir(source_root):
    person_path = os.path.join(source_root, person_id)
    if not os.path.isdir(person_path): continue

    for name in os.listdir(person_path):
        name_path = os.path.join(person_path, name)
        if not os.path.isdir(name_path): continue

        for quality in os.listdir(name_path):
            quality_path = os.path.join(name_path, quality)
            if not os.path.isdir(quality_path): continue

            images = [f for f in os.listdir(quality_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(images)
            split_idx = int(len(images) * train_ratio)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            for subset, subset_images in zip(["train", "val"], [train_images, val_images]):
                target_dir = os.path.join(output_root, subset, person_id, name, quality)
                os.makedirs(target_dir, exist_ok=True)
                for img in subset_images:
                    src = os.path.join(quality_path, img)
                    dst = os.path.join(target_dir, img)
                    shutil.copy2(src, dst)

print("✅ Dataset split completed.")
