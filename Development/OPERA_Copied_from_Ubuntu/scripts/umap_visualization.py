import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FEATURES_CSV = os.path.join(PROJECT_DIR, "features", "opera_features.csv")
FIGURES_DIR = os.path.join(PROJECT_DIR, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 데이터 로드
df = pd.read_csv(FEATURES_CSV)
df = df[df["label"] != "unknown"]

# 피처 및 레이블 분리
X = df.drop(columns=["filename", "label", "extraction_success"], errors="ignore").values
labels = df["label"].values

# UMAP 임베딩
reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# 시각화
plt.figure(figsize=(10, 8))
palette = sns.color_palette("hsv", len(np.unique(labels)))
sns.scatterplot(
    x=embedding[:, 0], y=embedding[:, 1],
    hue=labels, palette=palette, legend="full", s=10
)
plt.title("UMAP of OPERA Features")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(loc="best", title="Label")
fig_path = os.path.join(FIGURES_DIR, "umap_opera_features.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved UMAP visualization to: {fig_path}")
