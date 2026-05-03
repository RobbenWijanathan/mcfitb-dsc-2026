# Insurance Claims Forecasting

Predicting the **frequency, severity, and total amount** of health insurance claims using a weekly-aggregated time series ensemble approach. The pipeline covers the full lifecycle — from raw transaction data to multi-month forward projections.

---

## Project Structure

```
├── data/
│   ├── Data_Klaim.csv        # Claim transactions (Jan 2024 – Jul 2025)
│   └── Data_Polis.csv        # Active policyholder profiles
│
├── output/
│   ├── predictions_2025_2026.csv  # Final forecast output (Aug 2025–Dec 2026)
│   └── fig_*.png             # EDA and model evaluation plots
│
├── notebook.ipynb            # Main analysis notebook
└── README.md
```

---

## Overview

| Item | Detail |
|------|--------|
| **Targets** | Claim Frequency, Claim Severity, Total Claim Amount |
| **Forecast Horizon** | August 2025 – December 2026 (17 months) |
| **Granularity** | Weekly aggregation → monthly rollup |
| **Evaluation Metric** | MAPE (Mean Absolute Percentage Error) |
| **CV MAPE** | 6.22% *(walk-forward validation, 5-month test window)* |

---

## Methodology

### 1. Preprocessing
- Filter to paid claims only
- Aggregate to weekly frequency (83 data points)
- Winsorization at ±2.5σ to handle outliers
- Log transformation on total claim amounts

### 2. Modeling
Ten model architectures are trained and evaluated:

| Category | Models |
|----------|--------|
| Statistical | SES, ARIMA, Auto-SARIMA, Theta, Naïve, Seasonal Naïve |
| Deep Learning | NLinear, DLinear, MultiDLinear |
| Machine Learning | LightGBM |

### 3. Ensemble
- Weights assigned inversely proportional to each model's MAPE (1/MAPE weighting)
- Final forecast blends 80% ensemble output + 20% Seasonal Naïve

### 4. Evaluation
- Walk-forward (expanding window) cross-validation
- Test window: March – July 2025

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm scipy
```

### Running the Pipeline

1. Place `Data_Klaim.csv` and `Data_Polis.csv` in the `data/` directory.
2. Open and run `notebook.ipynb` from top to bottom.
3. Forecast results will be saved automatically to `output/predictions_2025_2026.csv`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation and aggregation |
| `matplotlib` | Visualization |
| `scikit-learn` | Preprocessing and evaluation utilities |
| `lightgbm` | Gradient boosting model |
| `scipy` | Statistical functions (winsorization, etc.) |

---

## Results

Forecast outputs are stored in `output/predictions_2025_2026.csv` with columns for predicted claim frequency, severity, and total amount at monthly granularity. Accompanying plots in `output/fig_*.png` cover exploratory data analysis, per-model performance, and ensemble diagnostics.