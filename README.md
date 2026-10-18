# Does Money Buy Wins? Predicting European Football Match Outcomes from Club Spending
**Traditional Machine Learning vs Deep Learning on Transfermarkt data**

*Ivan Olivier Muhoza*, African Leadership University, Machine Learning Summative

| Resource | Link |
|---|---|
| Report (PDF) | *add link* |
| Google Drive (raw data, model weights, TensorBoard logs) | *[link*](https://drive.google.com/drive/folders/1seN--4sPJiGN4XitmOG9qZvd0dKwJ4qU?usp=sharing) |
| Video presentation | *add link* |

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

## How to run
1. Open `football_prediction.ipynb` in Google Colab (GPU optional), or locally after `pip install -r requirements.txt`.
2. Run all cells top to bottom. Seed 42 is fixed throughout.
3. If the automatic download fails, download the dataset from Kaggle and unzip it into `data/player-scores/`.

## Outputs
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
