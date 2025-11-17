"""
Run a simple LightGBM baseline: load features, stratified split, train, evaluate, save model.
Usage:
    python scripts/run_baseline.py --features data/processed/train_features.parquet --output experiments/results/baseline
"""
import argparse
from pathlib import Path
import time


def main(features_path, output_dir, n_estimators, lr, test_size, random_state):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier
    import joblib

    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise SystemExit(f"Features not found: {features_path}")

    print(f"Loading features from {features_path} ...")
    df = pd.read_parquet(features_path)
    print(f"Loaded shape: {df.shape}")

    target_col = 'isDefault'
    if target_col not in df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in features")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print('Train shape:', X_train.shape, 'Val shape:', X_val.shape)

    clf = LGBMClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=random_state, n_jobs=-1)

    t0 = time.time()
    # Fit without early_stopping_rounds to maintain compatibility across lightgbm versions
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )
    t_end = time.time()

    pred_val = clf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, pred_val)
    print(f'Validation AUC: {auc:.6f}')
    print(f'Training time: {t_end - t0:.1f} s')

    model_path = output_dir / 'model.joblib'
    joblib.dump(clf, model_path)
    print(f'Saved model to {model_path}')

    # save a small oof sample to inspect
    oof_path = output_dir / 'val_preds.csv'
    import pandas as pd
    pd.DataFrame({'y_true': y_val.values, 'y_pred': pred_val}).to_csv(oof_path, index=False)
    print(f'Saved val preds to {oof_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='data/processed/train_features.parquet')
    parser.add_argument('--output', default='experiments/results/baseline')
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args.features, args.output, args.n_estimators, args.lr, args.test_size, args.random_state)
