from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> Tuple[KMeans, np.ndarray]:
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = model.fit_predict(X)
    return model, labels
