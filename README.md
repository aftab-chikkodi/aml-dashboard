# Adversarial ML Attack & Defense Dashboard

> CIA-3: Decision-Time Evasion Attacks and Defenses  
> Built with Streamlit + Plotly + scikit-learn

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Overview

An interactive dashboard demonstrating **white-box and black-box evasion attacks** against four ML classifiers, and evaluating five defense strategies on the **Wine dataset**.

### Models
| Model | Type |
|-------|------|
| Logistic Regression | Linear |
| SVM (RBF kernel) | Kernel-based |
| Random Forest | Ensemble |
| MLP | Neural Network |

### Attacks
| Attack | Type | Description |
|--------|------|-------------|
| WB1_FGSM | White-box | Fast Gradient Sign Method |
| WB2_PGD | White-box | Projected Gradient Descent |
| BB1_RandomSearch | Black-box | Random perturbation search |
| BB2_BoundaryAttack | Black-box | Decision boundary binary search |
| BB3_SurrogateTransfer | Black-box | Surrogate model transfer |

### Defenses
| Defense | Strategy |
|---------|----------|
| D1_AdversarialTraining | Augment training with adversarial examples |
| D2_DecisionRandomization | Majority vote with noise |
| D3_FeatureSqueezing | Bin/quantize input features |
| D4_ConfidenceRejection | Reject low-margin predictions |
| D5_RobustClipping | Clip inputs to training bounds |

## Dashboard Tabs

1. **Baseline** — Accuracy bars, confusion matrices, PCA projection
2. **Attack Analysis** — ASR heatmap, radar chart, grouped bar, sunburst
3. **Defense Analysis** — Before/after bars, reduction heatmap, lollipop chart
4. **ε Sweep** — Robustness curves, area chart, integrated vulnerability AUC
5. **Data Explorer** — Feature scatter, RF importances, correlation heatmap

## Run Locally

`ash
git clone https://github.com/YOUR_USERNAME/aml-dashboard.git
cd aml-dashboard
pip install -r requirements.txt
streamlit run app.py
`

## Deploy on Streamlit Cloud

1. Push to GitHub (steps below)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to pp.py
5. Click **Deploy**

## Tech Stack

- streamlit — Web framework
- plotly — Interactive visualizations
- scikit-learn — ML models, attacks & defenses
- 
umpy / pandas — Data processing
