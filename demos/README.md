# CaTrace Demo Notebooks

This folder contains demo notebooks to showcase how to use CaTrace for analyzing calcium imaging data.

#### `representational_similarity_one_vs_one.ipynb`
 
    - Representational similarity analysis (RSA) between a pair of synthetic manifolds.

    - Metrics used to quantify the similarity include:
      * Pearson correlation between mean activity patterns
      * Cosine distance between mean activity patterns
      * Euclidean distance between mean activity patterns (dE)
      * Generalized Mahalanobis distance between two manifolds (dM)

    - Useful for understanding metric behavior in isolation.

#### `representational_similarity_multiple_manifolds.ipynb`

    - Representational similarity analysis across multiple neural manifolds measured in zebrafish brain.

    - Helpful for deep-diving into how each metric behave in real datasets.