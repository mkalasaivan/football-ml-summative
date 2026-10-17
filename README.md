# Predicting Diabetes Risk from Self-Reported Health Indicators
**Traditional Machine Learning vs Deep Learning on the CDC BRFSS 2015 survey**

Henriette Biziyaremye, African Leadership University, Machine Learning Summative

| Resource | Link |
|---|---|
| Report (PDF) | *add link* |
| Google Drive (dataset, model weights, TensorBoard logs) | *add link* |
| Video presentation | *add link* |

## Problem
Many people with diabetes or prediabetes are undiagnosed because blood-glucose testing is not routine. This project tests whether cheap, self-reported indicators (blood pressure, cholesterol, BMI, general health, physical activity, income and others) can flag high-risk individuals so laboratory screening can be targeted.

## Dataset
CDC Diabetes Health Indicators, UCI Machine Learning Repository (ID 891), derived from the 2015 Behavioral Risk Factor Surveillance System. 253,680 respondents, 21 features, binary target `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes), about 14% positive. The notebook downloads it automatically with `ucimlrepo`. The raw data is **not** committed to this repository; a zipped copy is in the Google Drive folder.

## Approach
- **Preprocessing:** duplicate removal (to prevent train/test leakage), stratified 70/15/15 split, BMI outlier clipping with training-set bounds, standardisation of non-binary features.
- **Feature engineering:** cardiovascular risk count, unhealthy days, lifestyle score, WHO BMI bands, age × BMI interaction, care-access barrier, poor self-rated health.
- **Traditional ML (scikit-learn), 9 experiments:** Logistic Regression (baseline, class-weighted, engineered features, L1 feature selection), calibrated Linear SVM, Random Forest (default and tuned), Histogram Gradient Boosting (default and tuned). Tuning uses `RandomizedSearchCV` optimising average precision.
- **Deep learning (TensorFlow/Keras with `tf.data`), 8 experiments:** Sequential MLPs varying class weighting, depth, BatchNorm, dropout, L2, learning rate and batch size, plus a Functional-API wide-and-deep model with embeddings for ordinal features. Early stopping, learning-rate scheduling and TensorBoard logging throughout.
- **Evaluation:** PR-AUC (primary, given imbalance), ROC-AUC, precision, recall, F1, validation-tuned thresholds, confusion matrices, ROC/PR curves, learning curves, subgroup miss rates and permutation importance. The test set is used once, for the final models.

## How to run
1. Open `diabetes_prediction.ipynb` in Google Colab (GPU optional) or locally.
2. Locally: `pip install -r requirements.txt`.
3. Run all cells top to bottom. A fixed seed (42) is used throughout. Full run time is roughly 20–40 minutes on Colab.

## Outputs
| Path | Contents |
|---|---|
| `results/experiment_table.csv` / `.md` | Full experiment table (all 17 runs, validation metrics) |
| `results/test_results.csv` | Final test-set metrics |
| `results/*_search.csv` | Hyperparameter search results |
| `figures/` | EDA plots, learning curves, confusion matrices, ROC/PR curves, error analysis |
| `logs/` | TensorBoard logs for every run (`tensorboard --logdir logs`) |
| `models/` | Best traditional model (`.joblib`) and best deep model (`.keras`, `.h5`); uploaded to Drive, not committed |

## Repository structure
```
├── diabetes_prediction.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── figures/      (generated)
└── results/      (generated)
```

## Acknowledgements
Data: Centers for Disease Control and Prevention, BRFSS 2015, via the UCI Machine Learning Repository. Wide-and-deep architecture follows Cheng et al., "Wide & Deep Learning for Recommender Systems," 2016.
