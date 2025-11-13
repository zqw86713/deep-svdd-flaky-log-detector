import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# -------------------------
# Load embeddings + labels
# -------------------------
emb = np.load("data/processed/embeddings.npy")
labels = np.load("data/processed/labels.npy")  # 0 = normal, 1 = flaky-like

# -------------------------
# SimpleSVDD score function
# -------------------------
def svdd_score(x, center):
    return np.sum((x - center)**2, axis=1)

# -------------------------
# Estimate center from normal samples
# -------------------------
normal_emb = emb[labels == 0]
center = normal_emb.mean(axis=0)

# -------------------------
# Compute anomaly scores
# -------------------------
scores = svdd_score(emb, center)

# -------------------------
# ROC Curve
# -------------------------
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — SimpleSVDD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.show()

# -------------------------
# Precision–Recall Curve
# -------------------------
precision, recall, _ = precision_recall_curve(labels, scores)
plt.figure(figsize=(7,5))
plt.plot(recall, precision, label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — SimpleSVDD")
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=200)
plt.show()

# -------------------------
# Score Distribution
# -------------------------
plt.figure(figsize=(7,5))
plt.hist(scores[labels == 0], bins=20, alpha=0.7, label="Normal", color="blue")
plt.hist(scores[labels == 1], bins=20, alpha=0.7, label="Flaky-like", color="red")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.title("Score Distribution — Normal vs Flaky-like")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("score_distribution.png", dpi=200)
plt.show()

print("Done. Saved: roc_curve.png, pr_curve.png, score_distribution.png")
