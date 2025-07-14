import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import itertools

# === SETTINGS ===
embeddings_dir = "embeddings"
same_threshold = 0.5  # can be tuned later

# === LOAD ALL EMBEDDINGS ===
embeddings_dict = {}
for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        path = os.path.join(embeddings_dir, file)
        name = file.replace(".npy", "")
        emb = np.load(path)
        if emb.size > 0:
            embeddings_dict[name] = emb

# === GENERATE PAIRS ===
labels = []  # 1 for same person, 0 for different
similarities = []

names = list(embeddings_dict.keys())

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if j <= i:
            continue  # avoid duplicates and self-comparisons

        emb1 = np.mean(embeddings_dict[name1], axis=0).reshape(1, -1)
        emb2 = np.mean(embeddings_dict[name2], axis=0).reshape(1, -1)

        sim = cosine_similarity(emb1, emb2)[0][0]
        similarities.append(sim)

        label = 1 if name1.split('_')[0] == name2.split('_')[0] else 0
        labels.append(label)

# === Convert to arrays ===
similarities = np.array(similarities)
labels = np.array(labels)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# === Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(labels, similarities)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.savefig("precision_recall_curve.png")
plt.close()

# === Similarity Distribution ===
positive_sims = similarities[labels == 1]
negative_sims = similarities[labels == 0]

plt.figure(figsize=(8, 6))
plt.hist(positive_sims, bins=50, alpha=0.6, label='Same Person', color='green')
plt.hist(negative_sims, bins=50, alpha=0.6, label='Different People', color='red')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('Similarity Score Distribution')
plt.legend()
plt.savefig("similarity_distribution.png")
plt.close()

print("✅ All baseline evaluation plots saved:")
print("- roc_curve.png")
print("- precision_recall_curve.png")
print("- similarity_distribution.png")
