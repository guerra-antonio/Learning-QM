# Machine Learning approach to reconstruct Density Matrices from Quantum Marginals
[![DOI](https://img.shields.io/badge/DOI-10.1088%2F2632--2153%2Fade48d-blue)](https://iopscience.iop.org/article/10.1088/2632-2153/ade48d)

This repository presents a machine learning approach to address the **Quantum Marginals Problem**—a key question in quantum information theory and many-body physics. The problem revolves around determining a global system troughthout a set of reduced density matrices, describing subsystems of a quantum system.

## Problem Overview

The *Quantum Marginals Problem* arises when we want to reconstruct the global state of a quantum system based only on partial information. This partial information comes from the knowledge of smaller quantum states that describe subsystems of the global one. In particular, this project focuses on using machine learning techniques to achieve good results compared to conventional methods. Our goal is to determine whether a set of reduced k-body density matrices is consistent with an underlying global quantum state.

## Machine Learning Approach

This project builds a model that, given a non-positive matrix that's contain the information of the marginals, releases in the output a quantum state that describes correctly the marginals. The goal is to train the model to describe between consistent and inconsistent k-body density matrices for a global quantum state, offering an efficient alternative to traditional methods, which can be computationally expensive.

### The repository includes:
- **Data generation:** Scripts to generate valid and invalid sets of k-body marginals from known global quantum states.
- **Model training:** A machine learning model based on neural networks that relates "incompatible" matrices with "compatible" ones.
- **Evaluation and results:** Tools for evaluating the performance of the model, including fidelity and visualizations.

### BibTex 

```
@misc{arXiv:2410.11145,
      title={Machine Learning approach to reconstruct Density Matrices from Quantum Marginals}, 
      author={Daniel Uzcategui-Contreras and Antonio Guerra and Sebastian Niklitschek and Aldo Delgado},
      year={2024},
      eprint={2410.11145},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2410.11145}, 
}
```
