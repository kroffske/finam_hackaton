## 2025-10-04: Added news ticker extraction utilities
- Implemented ticker alias mapping and assignment helpers in src/finam/news_tickers.py
- Exposed helper functions from src/finam/__init__.py for pipeline reuse
## 2025-10-04: Added news preprocessing script
- Created scripts/0_news_preprocess.py to enrich news with ticker matches and configurable I/O paths
- Script supports optional exploded ticker outputs for downstream aggregation steps
- Extended script to compute and save ticker statistics (JSON + console output)
- Enhanced src/finam/news_tickers.py with NLTK-driven normalization and alias handling
- Added scripts/analyze_news_tickers.py for token frequency diagnostics on news data
- Updated scripts/0_news_preprocess.py default output to data/preprocessed_news and regenerated artifacts there
- Extended default news file list (train/public/private/test) with graceful skips if missing

## 2025-10-04: Implemented Cross-Validation for Time Series
- Created comprehensive docs/cross_validation.md documenting CV approach
  - Detailed analysis of trading days, gaps, and t+20 calculation
  - Explained data leakage risks and mitigation (gap = 21 days)
  - Provided practical examples and best practices
- Implemented src/finam/cv.py with core CV functions:
  - `get_trading_dates()` - extract sorted trading dates
  - `compute_t_plus_n()` - calculate t+N in trading days
  - `rolling_window_cv()` - main CV with gap protection
  - `evaluate_with_cv()` - convenient model evaluation with CV
- Updated README.md with CV documentation links and examples
- Key decisions:
  - Gap = 21 trading days (20 for t+20 + 1 safety margin)
  - Test size = 60 days (3 months of trading)
  - 5 folds for reliable evaluation

## 2025-10-04: Simplified project to focus on MAE only
- **Motivation & Problem Statement**:
  - **Original formulation** was complex: Score = 0.7×MAE_norm + 0.3×Brier_norm + 0.1×DA
  - Required 4 models: 2 regressors (return) + 2 classifiers (direction/probability)
  - **New formulation** is much simpler: **minimize only MAE** (Mean Absolute Error) for return prediction
  - Removed Brier Score (probability calibration metric) and Directional Accuracy (sign prediction)
  - Focus shifted entirely to accurate magnitude prediction of returns
  - This simplification removes the hardest-to-optimize components (Brier & DA) and focuses on core regression task
- **Code simplification (~40% reduction)**:
  - Removed `brier_score()`, `directional_accuracy()`, `normalized_score()` from src/finam/metrics.py
  - Simplified `evaluate_predictions()` to return only MAE metrics (mae_1d, mae_20d, mae_mean)
  - Removed classification models (prob_up_*) from src/finam/model.py
  - Deleted entire `CalibratedLightGBMModel` class (no calibration needed)
  - Simplified `LightGBMModel` to only 2 regressors (return_1d, return_20d) with MAE loss
  - Simplified `MomentumBaseline` (removed probability generation)
- **Scripts updated**:
  - scripts/2_train_model.py: removed --calibrate flag, y_direction_* targets, simplified evaluation
  - scripts/3_evaluate.py: removed confusion matrix, prob metrics, simplified reports
  - scripts/4_generate_submission.py: **added sigmoid fallback** to generate pred_prob_up_* from pred_return_* for submission compatibility
- **Documentation updated**:
  - CLAUDE.md: new "Iteration 6: Simplification for MAE" section
  - Updated task description, metrics formulas, code examples
  - Removed references to Brier/DA/normalized scores
- **Key benefits**:
  - ✅ Faster training (2 models instead of 4)
  - ✅ Clearer codebase (single metric focus)
  - ✅ Submission format preserved (sigmoid generates probabilities)
  - ✅ Same feature engineering and pipeline structure
