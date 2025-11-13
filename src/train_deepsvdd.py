# src/train_deepsvdd.py

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

from src.config import Config
from src.models.deep_svdd import SimpleSVDD


def load_embeddings_and_labels():
    emb_path = Config.PROCESSED_DIR / "embeddings.npy"
    label_path = Config.PROCESSED_DIR / "labels.npy"

    X = np.load(emb_path)   # shape: (N, D)
    y = np.load(label_path) # shape: (N,)

    print(f"Loaded embeddings: {X.shape}, labels: {y.shape}")
    return X, y


def train_simple_svdd():
    X, y = load_embeddings_and_labels()

    # 用 normal (label=0) 训练 SVDD
    X_normal = X[y == 0]
    print(f"Training SimpleSVDD on {X_normal.shape[0]} normal samples")

    # nu = 预期异常比例
    svdd = SimpleSVDD(nu=Config.SVDD_CONTAMINATION)

    svdd.fit(X_normal)

    # 保存模型
    out_path = Config.PROCESSED_DIR / "deep_svdd.pkl"
    joblib.dump(svdd, out_path)
    print(f"Saved SimpleSVDD model to {out_path}")

    # 评估
    evaluate_svdd(svdd, X, y)


def evaluate_svdd(svdd: SimpleSVDD, X: np.ndarray, y: np.ndarray):
    scores = svdd.decision_scores(X)  # 越大越异常

    # y: 0=normal, 1=flaky
    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)

    print("=== SimpleSVDD Evaluation ===")
    print(f"ROC AUC: {roc:.4f}")
    print(f"Average Precision (PR AUC): {ap:.4f}")

    normal_scores = scores[y == 0]
    flaky_scores = scores[y == 1]

    print(f"Mean score (normal): {normal_scores.mean():.4f}")
    print(f"Mean score (flaky):  {flaky_scores.mean():.4f}")

    print("Sample scores (first 10):")
    for i in range(min(10, len(scores))):
        print(f"  y={y[i]}, score={scores[i]:.4f}")


if __name__ == "__main__":
    train_simple_svdd()
