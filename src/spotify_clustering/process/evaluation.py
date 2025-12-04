from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .clustering import fit_kmeans


def evaluate_k_range(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, int]:
    
    raw_results = []
    best_k = None
    best_silhouette = -1.0 # Set the best silhouette to negative value

    # Go through k values to find the best one
    for k in k_values:
        print(f"Evaluating k={k}...")
        model, labels = fit_kmeans(X, n_clusters=k, random_state=random_state)
        inertia = model.inertia_
        silhouette = silhouette_score(
            X, 
            labels,
            sample_size=15000,
            random_state=random_state
        )

        raw_results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": silhouette,
            }
        )

        # Replace silhouette with better one if it exists
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    results = pd.DataFrame(raw_results)
    return results, best_k
