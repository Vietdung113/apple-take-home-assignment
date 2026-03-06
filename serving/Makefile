.PHONY: prepare-dataset eval-rouge eval-judge eval test

prepare-dataset:
	uv run python eval/prepare_dataset.py

eval-rouge: prepare-dataset
	uv run python eval/eval_rouge.py

eval-judge: prepare-dataset
	uv run python eval/eval_judge.py

eval: eval-rouge eval-judge

test:
	uv run python tests/test_e2e.py
