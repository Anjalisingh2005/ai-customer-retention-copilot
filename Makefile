.PHONY: install data train ingest pipeline api dashboard test docker-build docker-up clean

install:
	pip install -r requirements.txt

data:
	python scripts/generate_data.py

train:
	python scripts/train_model.py

ingest:
	python scripts/ingest_knowledge.py

pipeline:
	python scripts/run_pipeline.py --customer-id 7590-VHVEG

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/streamlit_app.py

test:
	pytest -q

docker-build:
	docker compose -f deployment/docker/docker-compose.yml build

docker-up:
	docker compose -f deployment/docker/docker-compose.yml up

clean:
	rm -rf artifacts/models/* artifacts/encoders/* artifacts/reports/* artifacts/faiss/* mlruns
