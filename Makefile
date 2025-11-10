# Basic Makefile for common tasks
.PHONY: setup data train predict clean

setup:
	@echo "(placeholder) setup environment, install requirements"

data:
	@echo "(placeholder) run data pipeline"
	python -m src.data.make_dataset --raw-dir data/raw --processed-dir data/processed

train:
	@echo "(placeholder) run training"
	python -m src.models.train --features data/processed/train_features.parquet --output-dir experiments/results/baseline

predict:
	@echo "(placeholder) run predict"
	python -m src.models.predict --model experiments/results/baseline/model.joblib --test data/processed/test.parquet --output submission.csv

clean:
	rm -rf experiments/results/* data/interim/* data/processed/*
