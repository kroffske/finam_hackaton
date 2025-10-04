# FINAL PLAYBOOK

## 0. Environment
- Python ≥3.10, LightGBM installed (`pip install lightgbm optuna`).
- Install project in editable mode: `pip install -e .` (loads `src/finam`).
- Raw CSVs live in `data/raw/participants/` as provided by the contest.

## 1. Data Prep
- Generate features & splits:
  ```bash
  python scripts/1_prepare_data.py --train-ratio 0.7 --val-ratio 0.15
  ```
- Output: `data/preprocessed/{train,val,test}.csv` + metadata with filtered feature list.

## 2. Hyperparameter Search (Optuna)
- Run Bayesian search over LightGBM params:
  ```bash
  python scripts/optuna_tune.py --n-trials 25 --study-name lightgbm_mae
  ```
- Results land in `outputs/<timestamp>_optuna/`:
  - `best_params.json` (for `2_train_model.py`)
  - `trials.csv` with every Optuna trial

## 3. Train Final Model
- Use tuned params on train/val split:
  ```bash
  python scripts/2_train_model.py --exp-name my_lgbm --model-type lightgbm \
      --n-estimators 350 --learning-rate 0.05 --num-leaves 64 \
      --min-child-samples 25 --subsample 0.8 --colsample-bytree 0.9
  ```
- Artifacts: `outputs/<timestamp>_<exp_name>/` with metrics, models, feature importance.

## 4. Full Dataset Fit (optional)
- Regenerate splits with zero validation (train+val merged):
  ```bash
  python scripts/1_prepare_data.py --train-ratio 0.85 --val-ratio 0.0
  ```
- Rerun `2_train_model.py` with best params to fit on maximum history.

## 5. Submission / Inference
- After final training, generate submission for latest experiment:
  ```bash
  python scripts/4_generate_submission.py --run-id <timestamp>_<exp_name>
  ```
- Outputs a submission CSV in the experiment folder.

## 6. Housekeeping
- Format/lint before commit: `ruff check --fix src/ scripts/ && ruff format src/ scripts/`.
- Log key actions in `SESSION.md` if continuing the hackathon log.
