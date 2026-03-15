## 26040079 - Data Science Competition 2026 — MCF ITB - hmm ywd gas

Prediksi frekuensi, severitas, dan total klaim asuransi kesehatan periode **Agustus 2025–Desember 2026** menggunakan pendekatan *ensemble* model deret waktu berbasis agregasi mingguan.

---

## Struktur Folder

```
├── data/
│   ├── Data_Klaim.csv        # 4,627 transaksi klaim (Jan 2024 – Jul 2025)
│   └── Data_Polis.csv        # Profil 4,096 pemegang polis aktif
│
├── output/
│   ├── predictions_2025_2026.csv  # Hasil prediksi final (Aug 2025–Dec 2026)
│   └── fig_*.png             # Visualisasi EDA dan evaluasi model
│
├── 26040079_DSC_Notebook_hmm_ywd_gas.ipynb   # Notebook utama
└── README.md
```

---

## Overview

| Item | Detail |
|------|--------|
| **Target** | Claim Frequency, Claim Severity, Total Claim |
| **Horizon** | Agustus 2025 – Desember 2026 (17 bulan) |
| **Granularitas** | Mingguan → rollup bulanan |
| **Metrik** | MAPE (Mean Absolute Percentage Error) |
| **CV MAPE** | 6.22% *(walk-forward, 5 bulan uji)* |

---

## Metodologi

1. **Preprocessing** — Filter klaim *paid*, agregasi mingguan (83 titik), *winsorization* ±2.5σ, transformasi log pada total klaim
2. **Model** — 10 arsitektur: SES, ARIMA, Auto-SARIMA, Theta, NLinear, DLinear, MultiDLinear, LightGBM, Naïve, Seasonal Naïve
3. **Ensemble** — Bobot 1/MAPE, blend 80% ensemble + 20% Seasonal Naïve
4. **Evaluasi** — Walk-forward validation (Mar–Jul 2025 sebagai jendela uji)

---

## Cara Menjalankan

1. Letakkan `Data_Klaim.csv` dan `Data_Polis.csv` di folder `data/`
2. Buka dan jalankan `26040079_DSC_Notebook_hmm_ywd_gas.ipynb`
3. Hasil prediksi tersimpan otomatis di `output/predictions_2025_2026.csv`

---

## Dependencies

```
numpy · pandas · matplotlib · scikit-learn · lightgbm · scipy
```
