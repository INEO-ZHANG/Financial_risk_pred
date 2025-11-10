"""
build_features.py
- 读取 data/processed，生成特征并写回 processed
- 这是占位脚本，请根据比赛字段实现具体特征工程
"""
import argparse
import pandas as pd
from pathlib import Path


def main(processed_dir):
    processed_dir = Path(processed_dir)
    train_path = processed_dir / 'train.parquet'
    if train_path.exists():
        df = pd.read_parquet(train_path)
        # TODO: 添加特征工程逻辑
        df.to_parquet(processed_dir / 'train_features.parquet', index=False)
        print(f"Wrote features to {processed_dir / 'train_features.parquet'}")
    else:
        print(f"No processed train found at {train_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data/processed')
    args = parser.parse_args()
    main(args.processed_dir)
