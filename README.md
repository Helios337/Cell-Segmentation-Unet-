# Cell Segmentation & Counting with Deep U-Net

A production-focused deep learning pipeline for segmenting and counting cell nuclei in biomedical images (BBBC038 / Data Science Bowl 2018), now with reproducible performance benchmarking, profiling, and regression guardrails.

## Features

- Deep U-Net training for semantic nuclei segmentation
- Watershed-based counting with **high_accuracy** and **fast** post-processing modes
- Reproducible training/evaluation artifacts (`training_summary.json`, `evaluation_metrics.json`)
- Multi-trial benchmark runner with variance and acceptance checks
- Profiling across data loading, preprocessing, training, inference, and post-processing
- Unit tests for metrics, model variants, loader integrity, and performance aggregation

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --model-variant baseline --epochs 20 --batch-size 16 --img-size 128
```

Optional performance controls:

```bash
python train.py --model-variant light --mixed-precision --max-samples 512 --output-dir runs/light_exp
```

### Model variants

- `baseline`: original quality-focused default
- `light`: faster separable-conv variant
- `tiny`: smallest/faster variant for constrained setups

## Evaluation

```bash
python evaluate.py --output-dir . --postprocess-mode high_accuracy
```

Fast mode:

```bash
python evaluate.py --output-dir . --postprocess-mode fast
```

Artifacts:

- `evaluation_report.png`
- `evaluation_metrics.json`

## Reproducible Benchmarking

Run frozen multi-trial baseline/candidate benchmarks:

```bash
python benchmark.py --trials 3 --epochs 20 --model-variant baseline --output-dir benchmark_runs/baseline
```

Candidate run:

```bash
python benchmark.py --trials 3 --epochs 20 --model-variant light --mixed-precision \
  --output-dir benchmark_runs/light \
  --baseline-report benchmark_runs/baseline/benchmark_report.json
```

Benchmark report includes:

- Hardware/software metadata
- Per-trial metrics and variance
- Prioritized bottlenecks by average wall-clock contribution
- Acceptance decision (quality + speed criteria)

## Acceptance Criteria (default)

Candidate must satisfy all:

- Mean IoU drop ≤ 0.01
- BinaryIoU drop ≤ 0.005
- Counting bias increase ≤ 1.0
- Train epoch speedup ≥ 1.10x
- Inference latency speedup ≥ 1.10x

## Tests

Run validation suite:

```bash
python -m unittest discover -s tests -v
```

Test coverage includes:

- IoU/counting correctness and edge cases
- Model output-shape correctness across architecture variants
- Data-loader dataset integrity and value-range checks
- Trial aggregation and acceptance-check logic

## CI Guardrails

- Quick smoke: unit tests on push/PR
- Periodic benchmark workflow: scheduled and manual execution for regression checks

## Notebooks

- `notebooks/results_analysis_plotly.ipynb`
  - Plotly graphs for baseline vs candidate comparison (speed, quality, bottlenecks, Pareto view).
- `notebooks/logical_changes_explained.ipynb`
  - Explains what changed and why, focused on logical design decisions.

To populate notebook charts, run benchmark artifacts first:

```bash
python benchmark.py --trials 1 --epochs 1 --model-variant baseline --max-samples 64 --output-dir benchmark_runs/baseline
python benchmark.py --trials 1 --epochs 1 --model-variant light --mixed-precision --max-samples 64 --output-dir benchmark_runs/candidate --baseline-report benchmark_runs/baseline/benchmark_report.json
```

## Core Files

- `train.py` – optimized training pipeline and profiling output
- `evaluate.py` – evaluation metrics, latency/throughput, count-agreement spread
- `benchmark.py` – reproducible multi-trial benchmarking and acceptance checks
- `performance.py` – benchmarking schema, aggregation, acceptance evaluation
- `utils.py` – IoU and post-processing modes

## License

Distributed under the MIT License. See `LICENSE` for more information.
