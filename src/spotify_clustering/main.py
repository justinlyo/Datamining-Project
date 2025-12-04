
# import config

import pandas as pd
from typing import Sequence
from pathlib import Path

from process.preprocessing import preprocess_for_clustering
from process.evaluation import evaluate_k_range
from process.clustering import fit_kmeans

PROJECT_ROOT = Path(".")
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_DATA_PATH = DATA_DIR / "processed" / "spotify_tracks_with_clusters.csv"
DATASET_PATH = DATA_DIR / "raw" / "spotify_tracks.csv"


RANDOM_STATE: int = 42
K_RANGE: Sequence[int] = range(2,15)

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "liveness",
    "valence",
    "tempo",
]

def main() -> None:

    # Read in dataset
    print(f"Loading raw data: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        print(f"Could not find file: {DATASET_PATH}")
        raise FileNotFoundError(f"Raw data file nto found at {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH)


    print(f"Preprocessing data for clustering")
    df_clean, X_scaled, _, used_features = preprocess_for_clustering(df_raw, FEATURE_COLUMNS)
    print(f"Using features: {used_features}")

    print(f"Evaluating K-Means for k in {K_RANGE}")
    results, best_k = evaluate_k_range(X_scaled, K_RANGE, RANDOM_STATE)
    print(f"Evaluation results:\n{results}")
    print(f"The best k by silhouette score is: {best_k}")

    print(f"Fitting the final K-Means model with k={best_k}")
    final_model, final_labels = fit_kmeans(
        X_scaled,
        n_clusters=best_k,
        random_state=RANDOM_STATE,
    )

    df_output = df_clean.copy()
    df_output["cluster"] = final_labels

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved final clustered dataset to {PROCESSED_DATA_PATH}")

    print(f"Complete!")


if __name__ == "__main__":
    main()
