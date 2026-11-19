PY ?= .venv/bin/python
SAMPLES ?= samples

.PHONY: setup test demo demo-synth api web clean

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -r requirements.txt
	uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu || uv pip install --python .venv torch
	uv pip install --python .venv -e .
	@echo "Also install Tesseract: sudo apt-get install tesseract-ocr"

test:
	$(PY) -m pytest -q

demo-synth:
	$(PY) -m reassemble.cli synth --out $(SAMPLES)

clean:
	rm -rf $(SAMPLES) result results

bench:
	$(PY) -m reassemble.cli bench

demo-solve: demo-synth
	$(PY) -m reassemble.cli solve $(SAMPLES)/memo_30_noisy --out result/memo_30_noisy

train:
	$(PY) -m reassemble.cli train

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

web:
	cd web && npm install && npm run dev
