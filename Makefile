.PHONY: install train test clean lint

VENV = venv
PYTHON = $(VENV)/bin/python

# ── Install ──────────────────────────────────────────────────
install: $(VENV)/.installed

$(VENV)/.installed: requirements.txt pyproject.toml
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	touch $(VENV)/.installed

# ── Training ─────────────────────────────────────────────────
train: install
	$(PYTHON) main.py

# ── Testing ──────────────────────────────────────────────────
test: install
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Quality ──────────────────────────────────────────────────
lint: install
	$(PYTHON) -m ruff check *.py

format: install
	$(PYTHON) -m ruff format *.py

# ── Clean ────────────────────────────────────────────────────
clean:
	rm -rf $(VENV) __pycache__ */__pycache__
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache *.egg-info
	rm -f best_model.h5