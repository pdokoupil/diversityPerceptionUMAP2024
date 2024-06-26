# About

This repository contains source code for paper entitled *User Perceptions of Diversity in Recommender Systems* presented at UMAP 2024 Conference.
Other supplementary material (e.g. pre-computed distance matrices, item-item matrices, etc.) including results can be found in [OSF repository](https://osf.io/9y8gx/).

# Contents

- [src/diversification.py](./src/diversification.py) contains source code for the diversification procedure and thin wrapper around relevance baseline (EASE^R) that can be loaded from pre-trained weights shared in the OSF repository
- [src/metrics.py](./src/metrics.py) contains implementations of all three metrics that were used for diversification (pretrained matrices can be found in OSF), together with additional metrics that were reported in the results (even though they were not directly used for diversification).

#

**Note:** there are fairly detailed comments in the source files themselves, including meaning of parameters and high-level idea of their expected usage.
We plan to release additional source codes, even source codes of the user study in the future (in upcoming ~months, most-likely late 2024 or early 2025).