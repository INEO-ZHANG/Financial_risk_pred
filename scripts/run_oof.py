"""
Run Stratified K-Fold LightGBM and produce OOF predictions and CV AUC.
Usage:
    python scripts/run_oof.py --features data/processed/train_features.parquet --output experiments/results/oof --folds 5
"""
import argparse
from pathlib import Path
import time
import json


def main(features_path, output_dir, folds, n_estimators, lr, random_state):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    import joblib

    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from {features_path} ...")
    df = pd.read_parquet(features_path)
    print(f"Loaded shape: {df.shape}")

    target_col = 'isDefault'
    if target_col not in df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in features")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    oof = np.zeros(len(df), dtype=float)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    fold_results = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{folds}: train {len(tr_idx)} val {len(val_idx)}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': lr,
            'verbose': -1,
            'seed': random_state,
        }

        bst = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train','valid']
        )

        val_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
        oof[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        print(f"Fold {fold+1} AUC: {auc:.6f}")

        # save model
        model_path = output_dir / f"lgbm_fold{fold+1}.txt"
        bst.save_model(str(model_path))

        fold_results.append({'fold': fold+1, 'auc': float(auc), 'best_iter': int(bst.best_iteration)})

    # overall
    cv_auc = roc_auc_score(y, oof)
    print(f"CV AUC ({folds}-fold): {cv_auc:.6f}")

    # save oof and results
    oof_path = output_dir / 'oof_preds.csv'
    pd.DataFrame({'y_true': y, 'y_pred': oof}).to_csv(oof_path, index=False)

    results = {'cv_auc': float(cv_auc), 'folds': fold_results}
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved oof to {oof_path}")
    print(f"Saved results to {output_dir / 'cv_results.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='data/processed/train_features.parquet')
    parser.add_argument('--output', default='experiments/results/oof')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args.features, args.output, args.folds, args.n_estimators, args.lr, args.random_state)
