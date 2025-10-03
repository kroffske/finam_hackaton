"""
Generate Submission Script

Этот скрипт:
1. Загружает обученную модель из outputs/<run_id>/
2. Загружает preprocessed public_test.parquet и private_test.parquet
3. Генерирует предсказания (pred_return_1d/20d, pred_prob_up_1d/20d)
4. Сохраняет submission файлы в outputs/<run_id>/

Usage:
    python scripts/4_generate_submission.py --run-id 2025-10-03_23-41-15_lgbm_with_news
    python scripts/4_generate_submission.py --run-id <run_id> --output-dir submissions/
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import joblib
import yaml


def generate_submission(
    run_id: str,
    output_dir: str = None,
    preprocessed_dir: str = None
):
    """
    Генерация submission файлов для public и private тестов

    Args:
        run_id: идентификатор эксперимента (например 2025-10-03_23-41-15_lgbm_with_news)
        output_dir: директория для сохранения submission (по умолчанию outputs/<run_id>/)
        preprocessed_dir: директория с preprocessed данными (по умолчанию data/preprocessed/)
    """
    print("=" * 80)
    print("GENERATE SUBMISSION")
    print("=" * 80 + "\n")

    # ========================================================================
    # 1. Проверка путей
    # ========================================================================
    print(f"[1/5] Loading experiment configuration...")

    exp_dir = project_root / 'outputs' / run_id

    if not exp_dir.exists():
        print(f"   ERROR: Experiment directory not found: {exp_dir}")
        print(f"\n   Available experiments:")
        outputs_dir = project_root / 'outputs'
        if outputs_dir.exists():
            for d in sorted(outputs_dir.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    print(f"      - {d.name}")
        return

    # Загружаем config
    config_path = exp_dir / 'config.yaml'
    if not config_path.exists():
        print(f"   ERROR: Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"   OK Loaded config from {config_path}")
    print(f"   Experiment: {config['experiment']['name']}")
    print(f"   Model type: {config['experiment']['model_type']}")

    # ========================================================================
    # 2. Загрузка модели
    # ========================================================================
    print(f"\n[2/5] Loading trained model...")

    # Ищем модель (может быть model_1d.pkl, model_20d.pkl или model.pkl)
    model_files = list(exp_dir.glob('model*.pkl'))

    if not model_files:
        print(f"   ERROR: No model files found in {exp_dir}")
        return

    # Загружаем модели
    models = {}
    for model_file in model_files:
        model_name = model_file.stem  # model_1d, model_20d, или model
        models[model_name] = joblib.load(model_file)
        print(f"   OK Loaded {model_file.name}")

    # ========================================================================
    # 3. Загрузка preprocessed данных
    # ========================================================================
    print(f"\n[3/5] Loading preprocessed test data...")

    if preprocessed_dir is None:
        preprocessed_dir = project_root / 'data' / 'preprocessed'
    else:
        preprocessed_dir = Path(preprocessed_dir)

    # Загружаем metadata для получения списка признаков
    metadata_path = preprocessed_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"   ERROR: metadata.json not found: {metadata_path}")
        return

    import json
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_columns']
    print(f"   OK Loaded {len(feature_cols)} features from metadata.json")

    # Загружаем public_test
    public_test_path = preprocessed_dir / 'public_test.parquet'
    private_test_path = preprocessed_dir / 'private_test.parquet'

    public_test_df = None
    private_test_df = None

    if public_test_path.exists():
        public_test_df = pd.read_parquet(public_test_path)
        print(f"   OK Loaded public_test.parquet: {len(public_test_df)} rows")
    else:
        print(f"   WARNING: public_test.parquet not found")

    if private_test_path.exists():
        private_test_df = pd.read_parquet(private_test_path)
        print(f"   OK Loaded private_test.parquet: {len(private_test_df)} rows")
    else:
        print(f"   WARNING: private_test.parquet not found")

    if public_test_df is None and private_test_df is None:
        print(f"\n   ERROR: No test data found!")
        print(f"   Run: python scripts/1_prepare_data.py")
        return

    # ========================================================================
    # 4. Генерация предсказаний
    # ========================================================================
    print(f"\n[4/5] Generating predictions...")

    # Определяем output директорию
    if output_dir is None:
        submission_dir = exp_dir
    else:
        submission_dir = Path(output_dir)
        submission_dir.mkdir(parents=True, exist_ok=True)

    def generate_predictions_for_test(test_df, test_name):
        """Генерация предсказаний для одного тестового набора"""
        if test_df is None:
            return None

        print(f"\n   Processing {test_name}...")

        # Подготовка данных
        X_test = test_df[feature_cols].fillna(0)

        # Проверяем какую модель используем
        if 'model' in models:
            # Единая модель для всех горизонтов
            model = models['model']
            predictions = model.predict(X_test)

        elif 'model_1d' in models and 'model_20d' in models:
            # Отдельные модели для 1d и 20d
            model_1d = models['model_1d']
            model_20d = models['model_20d']

            pred_1d = model_1d.predict(X_test)
            pred_20d = model_20d.predict(X_test)

            predictions = {
                'pred_return_1d': pred_1d['pred_return_1d'],
                'pred_prob_up_1d': pred_1d['pred_prob_up_1d'],
                'pred_return_20d': pred_20d['pred_return_20d'],
                'pred_prob_up_20d': pred_20d['pred_prob_up_20d']
            }

        elif all(k in models for k in ['model_return_1d', 'model_return_20d', 'model_prob_up_1d', 'model_prob_up_20d']):
            # Четыре отдельные модели (LightGBM с отдельными таргетами)
            import numpy as np

            # Предсказание доходностей
            pred_return_1d = models['model_return_1d'].predict(X_test)
            pred_return_20d = models['model_return_20d'].predict(X_test)

            # Предсказание вероятностей
            pred_prob_up_1d = models['model_prob_up_1d'].predict(X_test)
            pred_prob_up_20d = models['model_prob_up_20d'].predict(X_test)

            predictions = {
                'pred_return_1d': pred_return_1d,
                'pred_return_20d': pred_return_20d,
                'pred_prob_up_1d': pred_prob_up_1d,
                'pred_prob_up_20d': pred_prob_up_20d
            }

        else:
            print(f"   ERROR: Unknown model structure: {list(models.keys())}")
            return None

        # Создаем submission DataFrame
        submission = pd.DataFrame({
            'ticker': test_df['ticker'],
            'begin': test_df['begin'],
            'pred_return_1d': predictions['pred_return_1d'],
            'pred_return_20d': predictions['pred_return_20d'],
            'pred_prob_up_1d': predictions['pred_prob_up_1d'],
            'pred_prob_up_20d': predictions['pred_prob_up_20d']
        })

        print(f"      Generated {len(submission)} predictions")
        print(f"      Sample stats:")
        print(f"         pred_return_1d:  mean={submission['pred_return_1d'].mean():.6f}, std={submission['pred_return_1d'].std():.6f}")
        print(f"         pred_return_20d: mean={submission['pred_return_20d'].mean():.6f}, std={submission['pred_return_20d'].std():.6f}")
        print(f"         pred_prob_up_1d: mean={submission['pred_prob_up_1d'].mean():.4f}, std={submission['pred_prob_up_1d'].std():.4f}")
        print(f"         pred_prob_up_20d: mean={submission['pred_prob_up_20d'].mean():.4f}, std={submission['pred_prob_up_20d'].std():.4f}")

        return submission

    # Генерируем предсказания для public и private
    public_submission = generate_predictions_for_test(public_test_df, 'public_test')
    private_submission = generate_predictions_for_test(private_test_df, 'private_test')

    # ========================================================================
    # 5. Сохранение submission файлов
    # ========================================================================
    print(f"\n[5/5] Saving submission files...")

    if public_submission is not None:
        public_path = submission_dir / 'submission_public.csv'
        public_submission.to_csv(public_path, index=False)
        print(f"   OK Saved: {public_path}")
        print(f"      Rows: {len(public_submission)}")

    if private_submission is not None:
        private_path = submission_dir / 'submission_private.csv'
        private_submission.to_csv(private_path, index=False)
        print(f"   OK Saved: {private_path}")
        print(f"      Rows: {len(private_submission)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSION GENERATION COMPLETE!")
    print("=" * 80 + "\n")

    print("Summary:")
    print(f"   Experiment: {run_id}")

    if public_submission is not None:
        print(f"   Public submission:  {len(public_submission)} rows")
    if private_submission is not None:
        print(f"   Private submission: {len(private_submission)} rows")

    print(f"\n   Saved to: {submission_dir}/")

    print("\nNext steps:")
    if public_submission is not None:
        print(f"   # View public test predictions")
        print(f"   head {submission_dir / 'submission_public.csv'}")
    if private_submission is not None:
        print(f"   # View private test predictions")
        print(f"   head {submission_dir / 'submission_private.csv'}")

    print(f"\n   # Submit files to the competition platform")
    print(f"   # (Upload submission_public.csv and submission_private.csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate submission files from trained model')
    parser.add_argument('--run-id', type=str, required=True,
                        help='Experiment run ID (e.g., 2025-10-03_23-41-15_lgbm_with_news)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for submission files (default: outputs/<run_id>/)')
    parser.add_argument('--preprocessed-dir', type=str, default=None,
                        help='Directory with preprocessed data (default: data/preprocessed/)')

    args = parser.parse_args()

    generate_submission(
        run_id=args.run_id,
        output_dir=args.output_dir,
        preprocessed_dir=args.preprocessed_dir
    )
