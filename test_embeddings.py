import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# === Settings ===
embeddings_dir = "embeddings"
similarity_threshold = 0.5  # You can adjust based on ROC/AUC later

# === Load Embeddings ===
embeddings_dict = {}
for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        path = os.path.join(embeddings_dir, file)
        embeddings = np.load(path)
        if embeddings.size > 0:
            embeddings_dict[name] = embeddings
        else:
            print(f"⚠️ Empty embeddings: {file}")

# === Collect All Comparisons ===
true_labels = []
similarity_scores = []

names = list(embeddings_dict.keys())
print("\n🔍 Pairwise Similarity Report")
print("-" * 50)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name1 = names[i]
        name2 = names[j]
        emb1 = embeddings_dict[name1]
        emb2 = embeddings_dict[name2]

        # Average embedding for each identity
        avg_emb1 = np.mean(emb1, axis=0).reshape(1, -1)
        avg_emb2 = np.mean(emb2, axis=0).reshape(1, -1)

        # Compute cosine similarity
        similarity = cosine_similarity(avg_emb1, avg_emb2)[0][0]
        is_same = name1.split("_")[0] == name2.split("_")[0]  # True label

        # Store results
        similarity_scores.append(similarity)
        true_labels.append(1 if is_same else 0)

        result = "✅ MATCH" if similarity >= similarity_threshold else "❌ NO MATCH"
        print(f"{name1} vs {name2} → Similarity: {similarity:.4f} → {result}")

# === ROC Curve & AUC ===
fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], 'r--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Face Verification ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"\n📈 AUC Score: {roc_auc:.4f}")
print("🎉 Done! ROC curve saved as 'roc_curve.png'. All comparisons complete.")


