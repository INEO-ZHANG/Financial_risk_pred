"""
build_features.py
- 读取 data/processed，生成特征并写回 processed
- 这是占位脚本，请根据比赛字段实现具体特征工程
"""
import argparse
from pathlib import Path


def _freq_encode(df, col):
    """频次编码：返回 Series"""
    freq = df[col].value_counts(dropna=False)
    return df[col].map(freq).astype('int64')


def main(processed_dir, target_col='isDefault'):
    import pandas as pd

    processed_dir = Path(processed_dir)
    train_path = processed_dir / 'train.parquet'

    if not train_path.exists():
        print(f"No processed train found at {train_path}")
        return

    print(f"Loading {train_path} ...")
    df = pd.read_parquet(train_path)
    print(f"Loaded train with shape {df.shape}")

    # 分离目标（若存在）
    target = None
    if target_col in df.columns:
        target = df[target_col]

    # 简单特征工程策略：
    # - 数值列中位数填充
    # - 类别列频次编码
    # - 保留原始数值列和频次编码列

    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 移除 id 和目标列（不要当作特征）
    for c in ['id', target_col]:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    features = pd.DataFrame(index=df.index)

    # 数值列填充中位数
    for c in num_cols:
        median = df[c].median()
        features[c] = df[c].fillna(median)

    # 类别列频次编码
    for c in cat_cols:
        try:
            features[c + '_freq'] = _freq_encode(df, c)
        except Exception:
            # 如果映射失败，使用填充后的字符串长度作为简单特征
            features[c + '_len'] = df[c].fillna('').astype(str).map(len)

    # 若有 target，添加回 features 用于本地训练
    if target is not None:
        features[target_col] = target

    out_path = processed_dir / 'train_features.parquet'
    try:
        features.to_parquet(out_path, index=False)
        print(f"Wrote features to {out_path} (shape={features.shape})")
    except Exception:
        # 如果 parquet 不可用，回退为 csv
        csv_path = out_path.with_suffix('.csv')
        features.to_csv(csv_path, index=False)
        print(f"pyarrow/fastparquet not available, wrote features to {csv_path} (shape={features.shape})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--target-col', default='isDefault')
    args = parser.parse_args()
    main(args.processed_dir, target_col=args.target_col)
