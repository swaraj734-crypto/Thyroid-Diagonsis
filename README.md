# 🧬 Thyroid Diagnosis — Hybrid CatBoost + ANN Screening Tool

A lightweight clinical screening tool that predicts thyroid function
status (normal / hypothyroid / hyperthyroid) from standard lab
markers, using a **hybrid ensemble** of a CatBoost gradient-boosted
tree and a small feed-forward Artificial Neural Network (ANN).

Built as a learning project to explore how tree-based models and
neural networks can be combined — rather than picking just one — to
get a more stable prediction on a small, imbalanced medical dataset.

## Why a hybrid model?

CatBoost handles the tabular, mixed categorical/numeric data well and
is naturally robust to the kind of small, messy dataset you get in
medical screening. The ANN, on the other hand, can pick up softer,
non-linear interactions between markers (e.g. how TSH and FTI move
together) that a tree model might partially miss. Averaging their
predicted probabilities (`0.6 * CatBoost + 0.4 * ANN` by default)
gave more consistent results on hold-out data than either model
alone.

## How it works

```
Patient inputs (age, sex, TSH, T3, TT4, T4U, FTI, ...)
            │
            ▼
   ┌─────────────────────┐
   │  CatBoostClassifier  │──┐
   └─────────────────────┘  │   weighted
                             ├──► average ──► Final probability per class
   ┌─────────────────────┐  │
   │   Keras ANN model    │──┘
   └─────────────────────┘
```

## Project structure

```
thyroid-diagnosis/
├── app.py                 # Streamlit app (UI)
├── train_model.py         # Training script for the hybrid model
├── src/
│   ├── data_utils.py      # Loading + encoding helpers
│   └── hybrid_model.py    # CatBoost + ANN ensemble wrapper
├── data/
│   └── sample_thyroid_data.csv   # Small sample dataset (synthetic)
├── models/                # Trained model artifacts land here
└── requirements.txt
```

## Clinical markers used

| Marker | Meaning                      | Typical normal range |
|--------|-------------------------------|-----------------------|
| TSH    | Thyroid Stimulating Hormone   | 0.45 – 4.5 µIU/mL     |
| T3     | Triiodothyronine               | 0.8 – 2.0 nmol/L       |
| TT4    | Total T4                       | 60 – 180               |
| T4U    | T4 Uptake                      | 0.7 – 1.3              |
| FTI    | Free Thyroxine Index           | 65 – 150               |

## Getting started

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/thyroid-diagnosis.git
   cd thyroid-diagnosis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (uses the small sample dataset by default —
   swap in a larger dataset via `--data` once you have one)
   ```bash
   python train_model.py --data data/sample_thyroid_data.csv
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Notes on the dataset

The sample dataset included here is a small, synthetic set of
records shaped like the UCI Thyroid Disease dataset, meant only to
make the project runnable out of the box. For real experimentation,
swap in a larger labeled dataset with the same column names
(`age, sex, on_thyroxine, query_hypothyroid, TSH, T3, TT4, T4U, FTI, target`).

## Disclaimer

This project is for educational purposes only. It is **not** a
certified medical device and should never be used as a substitute
for diagnosis by a qualified healthcare professional.

---
Developed by **Shubham Swaraj** — B.Tech CSE (AI & DS), IIIT Senapati, Manipur
