# Nova Accretion Rate Reconstruction Using Machine Learning

Machine-learning investigation of whether accretion-rate histories can be reconstructed from luminosity evolution in recurrent nova simulations generated with MESA.

---

# Overview

Accretion-rate variability plays a fundamental role in determining the evolution and eruption properties of recurrent novae. While luminosity evolution can be obtained directly from simulations or observations, the underlying accretion-rate history is often difficult to infer.

This project explores whether machine-learning models can reconstruct the instantaneous accretion rate of an accreting white dwarf using only its luminosity history.

The study is based on a large suite of thermonuclear nova simulations generated with the MESA stellar evolution code. Multiple machine-learning algorithms were trained and compared, ranging from classical ensemble methods to modern deep-learning architectures.

This repository represents a follow-up machine-learning study built on top of simulation data originally produced during my Master's thesis research in computational stellar astrophysics.

---

# Scientific Motivation

One of the key challenges in nova studies is understanding how accretion histories influence eruption properties.

This project investigates the inverse problem:

**Luminosity History → Accretion Rate**

The goal is not simply to build a predictive model, but to explore how much information about the underlying accretion process is encoded within the luminosity evolution of the system.

---

# Dataset

The simulations were generated using MESA (Modules for Experiments in Stellar Astrophysics).

Simulation setup:

- White dwarf mass: 1.0 M☉
- Fixed stellar parameters
- Variable accretion histories
- Multiple nova cycles per simulation

Machine-learning dataset:

- 107 independent simulations
- Approximately 2,021,561 time samples
- Simulation-level train/test split
- Sliding-window time-series representation

Each row contains:

| Column | Description |
|----------|------------|
| time | Simulation time |
| logL | Logarithm of luminosity |
| log_mdot | Logarithm of accretion rate |
| sim_id | Simulation identifier |
| file_name | Source simulation file |

Example:

```csv
time,logL,log_mdot,sim_id,file_name
175.4221749,4.216476,-6.602855,0,nova_clean_binned_data_78.csv
175.4495534,4.216295,-6.602855,0,nova_clean_binned_data_78.csv
175.4769319,4.216124,-6.602855,0,nova_clean_binned_data_78.csv
```

---

# Methodology

The problem was formulated as a supervised regression task.

Input:

```text
Past luminosity history
```

Target:

```text
Current accretion rate
```

A sliding-window representation was used:

```text
[logL(t-W), ..., logL(t)]
            ↓
      log_mdot(t)
```

To ensure physically meaningful evaluation, the train-test split was performed at the simulation level rather than randomly across samples. This prevents information leakage between different portions of the same simulation.

---

# Window Size Experiments

A major objective of this work was to determine how much luminosity history is required to accurately reconstruct the accretion rate.

The following window sizes were explored:

- 50
- 100
- 200
- 400
- 600
- 800
- 1000
- 1200
- 1400

These experiments allowed us to investigate the importance of both short-term and long-term temporal information in nova light curves.

The final benchmark models primarily used a window size of 200, which provided the best overall balance between predictive performance and computational efficiency.

---

# Models Explored

## Tree-Based Ensemble Methods

- Random Forest
- XGBoost
- LightGBM
- CatBoost

## Deep Learning Models

- Multi-Layer Perceptrons (ANNs)
- 1D Convolutional Neural Networks (CNNs)
- Transformer-based sequence models
- Astroconformer-inspired architecture

---

# Physics-Informed Feature Engineering

Several feature representations were investigated.

### Luminosity Only

```text
logL
```

### Luminosity + First Derivative

```text
logL
+
d(logL)/dt
```

### Luminosity + First Derivative + Curvature

```text
logL
+
d(logL)/dt
+
d²(logL)/dt²
```

The first derivative proved particularly informative and significantly improved predictive performance.

---

# Results

| Model | R² Score |
|---------|---------:|
| Random Forest + d(logL)/dt | **0.877** |
| Random Forest | 0.849 |
| XGBoost | 0.847 |
| LightGBM | 0.847 |
| CatBoost | 0.836 |
| CNN | 0.829 |
| ANN | 0.814 |
| Astroconformer | 0.801 |
| RF + Derivative + Curvature | 0.755 |

---

# Best Model

### Random Forest + Luminosity Derivative

Performance metrics:

```text
RMSE = 0.0585
MAE  = 0.0269
R²   = 0.877
```

This model achieved the strongest performance among all tested approaches.

---

# Key Findings

### Luminosity contains substantial information about accretion history

Machine-learning models successfully recovered accretion-rate variations directly from luminosity evolution.

### Physics-informed features improve performance

Including the luminosity derivative produced the largest improvement in predictive accuracy.

### Extensive model benchmarking

The study includes a systematic comparison of:

- Ensemble tree methods
- Gradient-boosting methods
- Neural networks
- Convolutional architectures
- Transformer-based architectures

### Large-scale astrophysical dataset

The analysis was performed on more than two million samples generated from physically motivated stellar evolution simulations.

### Exploration of modern machine-learning approaches

Beyond traditional machine-learning methods, this project explored contemporary deep-learning architectures inspired by recent developments in astronomical time-series analysis.

---

# Repository Structure

```text
nova-accretion-rate-ml/

├── scripts/
│   ├── random_forest.py
│   ├── random_forest_derivative.py
│   ├── xgboost.py
│   ├── lightgbm.py
│   ├── catboost.py
│   ├── ann.py
│   ├── cnn.py
│   └── astroconformer.py
│
├── figures/
│   ├── model_comparison.png
│   ├── true_vs_predicted.png
│   └── residual_histogram.png
│
├── results/
│   ├── model_comparison,csv
|
├── docs/
│   └── project_report.pdf
|
└── README.md
```

# Related Research

The simulation dataset used in this repository originates from nova models developed during my Master's thesis research using MESA.

The original simulation work focused on accreting white dwarfs and thermonuclear nova outbursts. This repository extends that work by applying modern machine-learning techniques to investigate whether accretion histories can be inferred directly from luminosity evolution.

---

# Software and Tools

- Python
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- TensorFlow / Keras
- Matplotlib
- MESA

---

# Future Directions

Potential extensions include:

- Multiple white dwarf masses
- Variable metallicities
- Different core temperatures
- Uncertainty quantification
- Physics-informed machine learning
- Sequence-to-sequence forecasting
- Application to synthetic recurrent nova systems
- Application to observational light curves

---

# References

## MESA

Paxton, B., Bildsten, L., Dotter, A., et al. 2011, *Modules for Experiments in Stellar Astrophysics (MESA)*, ApJS, 192, 3.

Paxton, B., Cantiello, M., Arras, P., et al. 2013, ApJS, 208, 4.

Paxton, B., Marchant, P., Schwab, J., et al. 2015, ApJS, 220, 15.

Paxton, B., Schwab, J., Bauer, E. B., et al. 2018, ApJS, 234, 34.

Paxton, B., Smolec, R., Schwab, J., et al. 2019, ApJS, 243, 10.

---

## Nova Simulations and Accreting White Dwarfs

Yaron, O., Prialnik, D., Shara, M. M., & Kovetz, A. 2005, ApJ, 623, 398.

Wolf, W. M., Bildsten, L., Brooks, J., & Paxton, B. 2013, ApJ, 777, 136.

Hillman, Y., Prialnik, D., Kovetz, A., & Shara, M. M. 2016, ApJ, 819, 168.

Starrfield, S., Iliadis, C., & Hix, W. R. 2016, PASP, 128, 051001.

---

## Machine Learning

Breiman, L. 2001, *Random Forests*, Machine Learning, 45, 5–32.

Chen, T., & Guestrin, C. 2016, *XGBoost: A Scalable Tree Boosting System*.

Ke, G., Meng, Q., Finley, T., et al. 2017, *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*.

Prokhorenkova, L., Gusev, G., Vorobev, A., et al. 2018, *CatBoost: Unbiased Boosting with Categorical Features*.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. 1998, *Gradient-Based Learning Applied to Document Recognition*.

Vaswani, A., Shazeer, N., Parmar, N., et al. 2017, *Attention Is All You Need*.

---

## Astronomical Deep Learning

McLeod, A. F., Claytor, Z. R., Bellinger, E. P., et al. 2024, *Astroconformer: The Prospects of Analyzing Stellar Light Curves with Transformers*, The Astrophysical Journal.

---

# Author
Punit Dubey
M.Sc. Physics
NIT Rourkela, India

Research Interests:

- Machine Learning
- Computational Astrophysics
- Stellar Evolution
- Asteroseismology
- White Dwarfs
- Classical and Recurrent Novae
- Scientific Computing
- Time-Series Modeling
- Data-Driven Astronomy
- MESA
