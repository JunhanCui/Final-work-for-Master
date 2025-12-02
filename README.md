# Extended Persistence and Duality in Cubical Complexes

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-research-orange)
![TDA](https://img.shields.io/badge/topic-TDA%20%7C%20Medical%20Imaging-green)

## 📘 Overview

This repository serves as the **official computational implementation** for the Master's Thesis: **"Extended persistence and duality in cubical complexes"**.

The project applies **Topological Data Analysis (TDA)**, specifically **Cubical Homology**, to the problem of **Brain Tumor Classification** (Meningioma vs. Normal). By modeling medical images as cubical complexes, we extract robust topological invariants that persist across different filtration scales.

A key focus of this implementation is leveraging the interplay between **standard homology** and **duality** (via super-level set filtrations and periodic boundary conditions) to engineer high-dimensional features for machine learning.

## 🧠 Theoretical Foundation

The methodology is grounded in the algebraic topology of cubical sets. Unlike simplicial complexes used for point clouds, **Cubical Complexes** are the natural representation for pixel/voxel-based image data.

### 1. Extended Persistence & Duality
The code implements concepts derived from **Poincaré-Alexander Duality** in the context of persistence:
* **Sub-level Sets ($H_0$):** We analyze the evolution of connected components in the low-intensity regions.
* **Super-level Sets ($H_1$ via Duality):** The code explicitly handles **duality** by transforming $H_1$ diagrams (sign inversion of birth/death times). This corresponds to computing the homology of super-level sets (high-intensity tumor regions), utilizing the duality between the $k$-th homology of the object and the $(d-k-1)$-th cohomology of the background.

### 2. Toric Topology (`tcripser`)
We utilize **Toric Cubical Homology** (via `tcripser` with `embedded=True`) to handle boundary conditions. This allows us to distinguish between essential topological features (those wrapping around the "torus" of the image domain) and local geometric noise, providing a more robust filtration for medical imaging artifacts.

## ✨ Key Features

* **Cubical Homology Computation**: Utilizes `cripser` for standard homology and `tcripser` for toric/periodic homology.
* **Feature Engineering**:
    * **Total Persistence**: Quantifies the overall "significance" of topological features.
    * **Persistence Entropy**: A Shannon-entropy-based measure of the topological complexity and feature distribution.
* **Medical Image Pipeline**:
    * Support for **HDF5** 3D volume processing and slicing.
    * Automatic preprocessing of grayscale MRI scans.
    * **Dual-channel filtration**: Simultaneous extraction of $H_0$ (components) and $H_1$ (tunnels/loops).
* **Classification**: Logistic Regression model demonstrating the discriminative power of topological features alone.

## 📂 Repository Structure

This repository contains two versions of the implementation:

### 1. `tda_medical_pipeline.py` (Recommended)
**Status:** Refactored & Optimized
The production-ready version of the algorithm.
* **Features:** Modularized functions, vectorized entropy calculation, and robust exception handling.
* **Usage:** Run this file to execute the full classification pipeline.

### 2. `original_thesis_script.py`
**Status:** Legacy / Archival
The exact code used to generate the results reported in the Master's Thesis.
* Kept for reproducibility and historical reference.

## 🛠️ Installation

### Prerequisites
This project relies on a custom TDA module (`tcripser`) which requires **C++ Build Tools** to compile.

### Setup
```bash
# 1. Clone this repository
git clone [https://github.com/SanZhang/Extended-Persistence-Cubical.git](https://github.com/SanZhang/Extended-Persistence-Cubical.git)
cd Extended-Persistence-Cubical

# 2. Install dependencies
pip install -r requirements.txt