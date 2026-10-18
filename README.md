# Does Money Buy Wins? Predicting European Football Match Outcomes from Club Spending
**Traditional Machine Learning vs Deep Learning on Transfermarkt data**

*Ivan Olivier Muhoza*, African Leadership University, Machine Learning Summative

| Resource | Link |
|---|---|
| PDF report | [Football_ML_Summative_Report-Ivan-Olivier-Muhoza.pdf](https://github.com/user-attachments/files/33641522/Football_ML_Summative_Report-Ivan-Olivier-Muhoza.pdf) |
| Google Drive (raw data, model weights, TensorBoard logs) | [link](https://drive.google.com/drive/folders/1seN--4sPJiGN4XitmOG9qZvd0dKwJ4qU?usp=sharing) |
| Video presentation | [link](https://www.loom.com/share/4d336d268c5346deab7d737ea2b6e49) |

## Problem
Clubs like Manchester City have spent heavily on transfers, and their success is often credited to that money. This project measures how well financial power (squad market value and net transfer spend) predicts match results, and how much it adds beyond sporting signals such as Elo ratings and recent form. Each match is classified as **away win, draw or home win**. Champions League matches are analysed as a subset, with Manchester City as a case study.

## Dataset
*Football Data from Transfermarkt* (Kaggle: `davidcariboo/player-scores`), scraped from Transfermarkt.com. Tables used: `games`, `appearances`, `player_valuations`, `transfers`, `clubs`. The notebook downloads it with `kagglehub`. Raw data is **not** committed to this repository; a zipped copy is in the Google Drive folder.

## Approach
- **Leakage-free feature engineering:** every feature uses only information from before kick-off: rolling form (last 5 matches), Elo ratings, matchday squad market value (player valuations looked up as of the match date), and net/gross transfer spend over the previous 1 and 3 years.
- **Temporal split:** oldest 70% of matches for training, next 15% validation, most recent 15% test; `TimeSeriesSplit` for hyperparameter search.
- **Traditional ML (scikit-learn), 8 experiments:** a money-only vs sport-only vs combined ablation with Logistic Regression, class weighting, and default vs tuned Random Forest and Histogram Gradient Boosting.
- **Deep learning (TensorFlow/Keras with `tf.data`), 8 experiments:** Sequential MLPs varying class weights, depth, BatchNorm, dropout, L2, learning rate and batch size, plus a Functional-API model with a learned competition embedding.
- **Evaluation:** macro ROC-AUC (primary), log loss, accuracy, macro F1, draw recall, confusion matrices, one-vs-rest ROC curves, learning curves, accuracy by squad-value gap, permutation importance, UCL subset and Manchester City case study.

## Results
57,529 league and European matches (2012–2026) after filtering; test period May 2024 to May 2026 (8,630 matches).

| Model (test set) | Accuracy | Macro ROC-AUC | Log loss | Macro F1 |
|---|---|---|---|---|
| ML-6 Tuned Random Forest (best traditional) | **0.534** | **0.673** | **0.976** | 0.399 |
| DL-5 MLP, learning rate 1e-2 (best deep) | 0.532 | 0.672 | 0.976 | 0.398 |
| ML-1 Logistic Regression, money features only | 0.526 | 0.667 | 0.982 | 0.390 |
| Rule: richer squad wins | 0.516 | – | – | 0.392 |
| Rule: always home win | 0.435 | – | – | 0.202 |

Key findings:
- **Squad value, not transfer spend, predicts results.** Squad value correlates with points per game (Spearman ρ = 0.46) while 12-month net spend does not (ρ = 0.00). Squad-value difference is the most important feature in the best model.
- **Money and sport are largely redundant.** Money-only and sport-only models score almost the same (validation ROC-AUC 0.674 vs 0.677); combining them gives 0.683.
- **Deep learning matched but did not beat traditional ML.** All tuned models sit within about 0.003 ROC-AUC of each other, so the limit is the information in the features, not model capacity.
- **Money matters most when the gap is large.** Accuracy rises from 44% in the most evenly valued fifth of matches to 72% in the most lopsided fifth; upsets fall from 33% to 11%.
- **Draws are the hard class.** Unweighted models never predict a draw; class weights raise draw recall to about 0.23–0.24 at a cost in accuracy and log loss.
- **Champions League matches are easier to predict** than league matches (59.8% vs 53.1% accuracy, 378 test matches).

See the notebook and report for the full experiment table, error analysis and the Manchester City case study.

## Repository contents
| File | Purpose |
|---|---|
| `football_prediction.ipynb` | Full pipeline: data loading, feature engineering, EDA, 16 experiments, test evaluation, error analysis |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps raw data, model weights, logs and the virtual environment out of the repository |

## How to run
**Google Colab (simplest):** open `football_prediction.ipynb` in Colab (GPU optional) and run all cells top to bottom.

**Locally:** TensorFlow does not yet support Python 3.13+, so use Python 3.10–3.12.
```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook football_prediction.ipynb
```

Notes:
- Seed 42 is fixed throughout for reproducibility.
- The dataset downloads automatically through `kagglehub`. If that fails, download it from Kaggle and unzip it into `data/player-scores/`.
- The hyperparameter searches and 8 neural networks take a while on CPU; a Colab GPU speeds up the deep learning section.

## Outputs
Running the notebook creates the folders below. `data/`, `models/` and `logs/` are git-ignored and shared through the Google Drive folder instead.


| Path | Contents |
|---|---|
| `results/experiment_table.csv` / `.md` | All 16 experiments with settings and validation metrics |
| `results/test_results.csv` | Final test metrics, including simple baselines |
| `results/*_search.csv` | Hyperparameter search results |
| `results/accuracy_by_value_gap.csv`, `results/ucl_subset_results.csv` | Error analysis tables |
| `figures/` | EDA, learning curves, confusion matrices, ROC curves, error analysis, Man City case study |
| `logs/` | TensorBoard logs for every run (`tensorboard --logdir logs`) |
| `models/` | Best traditional (`.joblib`) and deep (`.keras`, `.h5`) models; uploaded to Drive, not committed |

## Acknowledgements
Data from Transfermarkt via the `transfermarkt-datasets` project by David Cariboo (Kaggle). Elo rating method after Elo (1978) as adapted for football.
