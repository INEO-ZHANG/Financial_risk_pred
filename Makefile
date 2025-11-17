# Basic Makefile for common tasks
.PHONY: setup data train predict clean

setup:
	@echo "(placeholder) setup environment, install requirements"

data:
	@echo "(placeholder) run data pipeline"
	python -m src.data.make_dataset --raw-dir data/raw --processed-dir data/processed

train:
	@echo "Run training baseline & OOF"
	python scripts/run_baseline.py --features data/processed/train_features.parquet --output experiments/results/baseline
	python scripts/run_oof.py --features data/processed/train_features.parquet --output experiments/results/oof

predict:
	@echo "Run test prediction"
	python scripts/predict_testA.py --test-raw data/raw/testA.csv --train-processed data/processed/train.parquet --train-features data/processed/train_features.parquet --model-dir experiments/results/oof --sample data/raw/sample_submit.csv --output submission.csv

clean:
	rm -rf experiments/results/* data/interim/* data/processed/*
