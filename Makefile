# Optional convenience targets
.PHONY: install data baselines multimodal ablation evaluate explain test api ui

install:
	pip install -e ".[dev,boost]"

data:
	python scripts/prepare_dataset.py --source demo --n-samples 120

baselines:
	python scripts/train_baseline.py --config configs/baseline.yaml

multimodal:
	python scripts/train_multimodal.py --config configs/multimodal.yaml

ablation:
	python scripts/run_ablation.py --config configs/multimodal.yaml --epochs 8

evaluate:
	python scripts/evaluate.py --checkpoint outputs/models/multimodal_best.pt

explain:
	python scripts/explain.py --checkpoint outputs/models/multimodal_best.pt

test:
	ruff check .
	pytest -q -m "not slow"
	mypy src/materials_ai --ignore-missing-imports

api:
	uvicorn materials_ai.api.main:app --reload --port 8000

ui:
	streamlit run app/streamlit_app.py
