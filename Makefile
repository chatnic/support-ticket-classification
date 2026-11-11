.PHONY: setup data train test llm-eval all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --n 12000 --seed 42 --out data/raw/tickets.csv

train:
	PYTHONPATH=src python -m ticketclf.pipeline --data data/raw/tickets.csv --out reports

test:
	python -m pytest -q

# Optional: needs ANTHROPIC_API_KEY. Scores 400 never-seen-phrasing tickets with an LLM.
llm-eval:
	PYTHONPATH=src python -m ticketclf.llm --sample 400

all: data train test
