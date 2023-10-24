# GALA: Graph Diffusion-based Alignment with Jigsaw for Source-free Domain Adaptation

## Abstract

Source-free domain adaptation is a crucial machine learning topic, as it contains numerous applications in the real world, particularly with respect to data privacy. Existing approaches predominantly focus on Euclidean data, such as images and videos, while the exploration of non-Euclidean graph data remains scarce. Recent graph neural network (GNN) approaches could suffer from serious performance decline due to domain shift and label scarcity in source-free adaptation scenarios. In this study, we propose a novel method named Graph Diffusion-based Alignment with Jigsaw (GALA) tailored for source-free graph domain adaptation. To achieve domain alignment, GALA employs a graph diffusion model to reconstruct source-style graphs from target data. Specifically, a score-based graph diffusion model is trained using source graphs to learn the generative source styles. Then, we introduce perturbations to target graphs via a stochastic differential equation instead of sampling from a prior, followed by the reverse process to reconstruct source-style graphs. We feed them into an off-the-shelf GNN and introduce class-specific thresholds with curriculum learning, which can generate accurate and unbiased pseudo-labels for target graphs. Moreover, we develop a simple yet effective graph mixing strategy named graph jigsaw to combine confident graphs and unconfident graphs, which can enhance generalization capabilities and robustness via consistency learning. Extensive experiments on benchmark datasets validate the effectiveness of GALA.


## Code Usage

python main.py --config configs/ENZYMES_0.py --mode train --workdir exp/ENZYMES


## Acknowledgement

- This repo is based on the repo [Score_SDE](https://github.com/yang-song/score_sde_pytorch) and [GraphGDP](https://github.com/GRAPH-0/GraphGDP). 

- Evaluation implementation is modified from the repo [GGM-metrics](https://github.com/uoguelph-mlrg/GGM-metrics).
