.PHONY: setup quality train app test all clean

setup:
	pip install -r requirements.txt

quality:
	python src/data_quality.py

train:
	python src/train.py

app:
	streamlit run app/app.py

test:
	pytest tests/ -v

all: setup quality train

clean:
	rm -rf models/*.pkl models/*.json data/processed/*.csv data/processed/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
