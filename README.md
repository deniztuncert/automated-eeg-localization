# 🧠 Automated 3D EEG Electrode Identification & Labeling Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced pipeline developed to automate the identification and labeling of EEG electrodes from 3D head scans. This project bridges the gap between **clinical anatomy** and **computational neuroengineering**.

---

## 📸 Project Showcase
> <img width="1470" height="956" alt="Screenshot 2026-01-02 at 22 11 28" src="https://github.com/user-attachments/assets/e66e177f-9f18-4a92-9e4c-1e70bed084dd" />

---

## 🛠 Features
* **Geometry-Based Extraction:** Utilizes **Gaussian Curvature** analysis to isolate electrode sockets from anatomical structures.
* **Intelligent Clustering:** Employs **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) to accurately group points into discrete electrodes.
* **Automated Labeling:** Maps identified coordinates to the **International 10-20 system** using Affine Transformations.
* **Neuroimaging Standards:** Fully compatible with the **RAS (Right-Anterior-Superior)** coordinate system.
* **Interoperability:** Native export support for `.locs` and `.elp` formats (compatible with **EEGLAB, MNE-Python, and FieldTrip**).

---

## 🧪 Methodology

The pipeline follows a 4-step computational process:

1.  **Preprocessing:** Mesh normalization and Z-axis anatomical clipping.
2.  **Feature Detection:** Curvature thresholding to identify potential electrode sites.
3.  **Spatial Clustering:** DBSCAN-based noise reduction and hub identification.
4.  **Template Registration:** Least-Squares fitting of the subject's points to the 10-20 template.

> <img width="1470" height="956" alt="Screenshot 2026-01-02 at 22 14 19" src="https://github.com/user-attachments/assets/db830c21-5592-418d-8780-5abbe8bb0a9e" />

---

## 📦 Data Optimization & GitHub Constraints
Due to GitHub's **25MB file size limit** for direct uploads, the sample data in this repository has been optimized:
* **Anatomical Clipping:** The provided `.ply` mesh is intentionally clipped below the upper cranium. This reduces file size while preserving all necessary electrode sites and landmarks (Nz, Iz, LPA, RPA).
* **Binary PLY Format:** Original scans were converted to Binary PLY to maintain high vertex precision for curvature calculations without the overhead of ASCII formats.

---
## 🚀 Installation & Usage
1. **Clone the repository:**
   `git clone https://github.com/deniztuncert/automated-eeg-localization`

2. **Install dependencies:**
   `pip install -r requirements.txt`

3. **Run the pipeline:**
   `python EEG_AutoElec_Detection_01.py`

---

## 📈 Roadmap (v2.0)

* [ ] **Hungarian Algorithm:** Implementation for global cost minimization in label assignment.
* [ ] **Rigid Transformation + Isotropic Scaling:** Maintaining geometric proportions during warping.
* [ ] **Auto-Landmark Detection:** Deep learning-based Nasion/Inion estimation.
* [ ] **BEM Model Integration:** Automated Boundary Element Method head model generation.

---

## 👨‍🔬 Author

**Deniz Tuncer Tepe**  
First-year medical student with a research focus on neuroscience and neurotechnology.

**Research interests:**  
EEG source localization, biosignal analysis, 3D neuroimaging pipelines, and AI-driven biomedical systems.
