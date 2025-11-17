"""
Predict on testA.csv using K-fold LightGBM models (lgbm_fold*.txt) and produce submission matching sample_submit.csv.
"""
import argparse
from pathlib import Path


def main(test_raw_path, train_processed_path, train_features_path, model_dir, sample_submit_path, output_path):
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from collections import defaultdict

    test_raw_path = Path(test_raw_path)
    train_processed_path = Path(train_processed_path)
    train_features_path = Path(train_features_path)
    model_dir = Path(model_dir)
    sample_submit_path = Path(sample_submit_path)
    output_path = Path(output_path)

    if not test_raw_path.exists():
        raise SystemExit(f"Test file not found: {test_raw_path}")
    if not train_processed_path.exists():
        raise SystemExit(f"Train processed not found: {train_processed_path}")
    if not train_features_path.exists():
        raise SystemExit(f"Train features not found: {train_features_path}")
    if not sample_submit_path.exists():
        raise SystemExit(f"Sample submission not found: {sample_submit_path}")

    print('Loading train processed and features to get mappings...')
    train_df = pd.read_parquet(train_processed_path)
    train_feat_df = pd.read_parquet(train_features_path)

    # feature columns used by model (exclude target)
    if 'isDefault' in train_feat_df.columns:
        feature_cols = [c for c in train_feat_df.columns if c != 'isDefault']
    else:
        feature_cols = train_feat_df.columns.tolist()

    print('Feature columns count:', len(feature_cols))

    # derive numeric medians from train_df (original numeric cols)
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # remove id
    if 'id' in num_cols:
        num_cols.remove('id')
    if 'id' in cat_cols:
        cat_cols.remove('id')

    medians = {c: train_df[c].median() for c in num_cols}
    # frequency maps for categorical
    freq_maps = {c: train_df[c].value_counts(dropna=False).to_dict() for c in cat_cols}

    # load test raw
    print('Loading test raw...')
    test_df = pd.read_csv(test_raw_path)

    # build features similar to build_features.py
    features = pd.DataFrame(index=test_df.index)

    # numeric
    for c in num_cols:
        if c in test_df.columns:
            features[c] = test_df[c].fillna(medians.get(c, 0))
        else:
            # if column missing in test, fill with median or 0
            features[c] = medians.get(c, 0)

    # categorical -> freq encoding
    for c in cat_cols:
        col_name = c + '_freq'
        if c in test_df.columns:
            fmap = freq_maps.get(c, {})
            features[col_name] = test_df[c].map(lambda x: fmap.get(x, 0)).astype('int64')
        else:
            features[col_name] = 0

    # ensure feature_cols exist in features; if some numeric features were not in num_cols (e.g., remained as freq cols), handle
    for c in feature_cols:
        if c not in features.columns:
            # try to create from train_feat_df (mean)
            if c in train_feat_df.columns:
                features[c] = train_feat_df[c].mean()
            else:
                features[c] = 0

    # reorder
    features = features[feature_cols]
    print('Final test features shape:', features.shape)

    # load models
    model_files = sorted(model_dir.glob('lgbm_fold*.txt'))
    if not model_files:
        # fallback to single baseline model
        single = model_dir / 'model.joblib'
        if single.exists():
            import joblib
            print('Loading single model', single)
            clf = joblib.load(single)
            preds = clf.predict_proba(features)[:,1]
        else:
            raise SystemExit('No models found in model_dir')
    else:
        print('Found models:', model_files)
        preds_all = []
        for mf in model_files:
            bst = lgb.Booster(model_file=str(mf))
            preds_all.append(bst.predict(features, num_iteration=bst.best_iteration))
        preds = np.mean(preds_all, axis=0)

    # prepare submission
    sub = pd.read_csv(sample_submit_path)
    # assume id column present and order matches sample
    if 'id' in sub.columns:
        sub_ids = sub['id']
        # align by order: we assume test_df order matches sample
        # otherwise match by id column if test has id
        if 'id' in test_df.columns:
            # ensure same order as sample
            test_map = pd.Series(preds, index=test_df['id']).to_dict()
            sub['isDefault'] = sub['id'].map(lambda x: test_map.get(x, 0.0))
        else:
            sub['isDefault'] = preds
    else:
        # fallback: append probability column
        sub['isDefault'] = preds

    sub.to_csv(output_path, index=False)
    print('Wrote submission to', output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-raw', default='data/raw/testA.csv')
    parser.add_argument('--train-processed', default='data/processed/train.parquet')
    parser.add_argument('--train-features', default='data/processed/train_features.parquet')
    parser.add_argument('--model-dir', default='experiments/results/oof')
    parser.add_argument('--sample', default='data/raw/sample_submit.csv')
    parser.add_argument('--output', default='submission.csv')
    args = parser.parse_args()
    main(args.test_raw, args.train_processed, args.train_features, args.model_dir, args.sample, args.output)
