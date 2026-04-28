<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="pictures/catrace_small_flipped.png" width="15%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# CaTrace

<em></em>

</div>
<br>
<strong>CaTrace</strong> is a toolbox for analyzing calcium imaging time traces of neuronal population.
<br>
It offers tools for basic neuronal response analysis, including response amplitude quantification, response pattern similarity metrics.
<br>
More advanced analysis including distance between neural manifolds using Euclidean and Mahalanobis distance is included.
<br>
Statistical analysis tools are provided to detect difference between experimental groups.

---

## Overview
<img src="pictures/catrace_overview.png" width="95%" style="position: relative; top: 0; right: 0;" alt="CaTrace Project Overview"/>

---
## Getting Started

### Prerequisites

The dependent python packages will automatically be installed by pip. Here a few dependent package is highlighted for important features:
- **Cascade**: Cascade is a toolbox by **[Rupprecht et al. 2021]** that translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes using deep learning models. CaTrace depends on it to deconvolve the ΔF/F time traces.
- **GLUE**: This is a package by **[Chou et al. 2024]** for manifold capacity analysis. The connector to this package will be accessible after the official release of GLUE.

### Installation
#### Default installation
Install CaTrace:
```
pip install git+https://github.com/aloejhb/catrace
```
or clone the repository locally and install with
```
git clone git@github.com:aloejhb/catrace
cd catrace; pip install -e .
```
#### Installation with spike inference feature
To use [Cascade](https://github.com/HelmchenLabSoftware/Cascade) for spike inference, you need to first install the depedency specified in `requirements-spikeinf.txt`
```
pip install -r requirements-spikeinf.txt
```
Then install CaTrace with spike inference enabled:
```
pip install "catrace[spikeinf] @ git+https://github.com/aloejhb/catrace.git"
```
or clone the repository locally and install with
```
git clone git@github.com:aloejhb/catrace
cd catrace; pip install -e .[spikeinf]
```
Note that currently in requirements-spikeinf.txt, a forked version of original Cascade is specified, to ensure compatibility with CaTrace. If you use Cascade for your paper, you can cite the original publication by **[Rupprecht et al. 2021]**.

### Usage
You can use CaTrace to run analysis on a single experiment or batch your analysis across a set of experiments.

Refer to the [demos](demos/) for examples.

The jupyter notebooks generating figures for the Hu, Temiz et al. paper can be found at [pDp_manifold_analysis](https://github.com/aloejhb/pDp_manifold_analysis).
---

## Features

| Feature Category                 | Feature                                                                         | 
|----------------------------------|---------------------------------------------------------------------------------|
| **Preprocessing**                | Read raw fluorescence traces                                                    |
|                                  | Compute ΔF/F traces                                                             |
|                                  | Deconvolve to infer spike probability and spike rate using [Cascade](https://github.com/HelmchenLabSoftware/Cascade) |
| **Response amplitude**           | Compute neuron response during a time window                                    |
| **Pattern similarity**           | Measure the similarity between response patterns, including pattern correlation, cosine distance | 
| **Distance between manifolds**   | Compute distance between manifolds, including Euclidean distance between manifold centers (dE) and Mahalanobis distance (dM) | 
| **Dimensionality reduction**     | Reduce dimensionality of neuronal population activity patterns, including PCA, isomap, UMAP etc., using scikit-learn              |
| **Statistics and plotting**      | Perform statistical tests on experimental group data, including Mann-Whitney U, Kruskal-Wallis, t-test and plot relevant plots|
| **Correlation with behavior**    | Correlate geometric properties of neuronal representations with behavior parameters, e.g. behavior discrimination score           |



---

## References
1. Hu, B., Temiz, N. et al. Representational learning by optimization of neural manifolds in an olfactory memory network. 2024.11.17.623906 Preprint at https://doi.org/10.1101/2024.11.17.623906 (2024).
2. Rupprecht, P. et al. A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. Nat Neurosci 24, 1324–1337 (2021).
3. Chou, C.-N. et al. Neural Manifold Capacity Captures Representation Geometry, Correlations, and Task-Efficiency Across Species and Behaviors. 2024.02.26.582157 Preprint at https://doi.org/10.1101/2024.02.26.582157 (2024).
---

## License

Catrace is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---


[def]: demos\
