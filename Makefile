PY ?= .venv/bin/python
SAMPLES ?= samples

.PHONY: setup test demo demo-synth api web clean

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -r requirements.txt
	uv pip install --python .venv -e .
	@echo "Also install Tesseract: sudo apt-get install tesseract-ocr"

test:
	$(PY) -m pytest -q

demo-synth:
	$(PY) -m reassemble.cli synth --out $(SAMPLES)

clean:
	rm -rf $(SAMPLES) result results
