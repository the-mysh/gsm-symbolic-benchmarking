"""Benchmark orchestration for GSM-Symbolic evaluation.

This subpackage provides the core benchmarking infrastructure:

* `BenchmarkConfig`: Configuration dataclass for benchmark settings, memory
  allocation, and model loading parameters.
* `BenchmarkRunner`: Main orchestrator that runs multi-model evaluation across
  dataset variants.
* `ModelEvaluator`: Wraps model inference and answer extraction for a single
  model.
* `AnswerExtractor`: Extracts numeric answers from model outputs (textual or
  code-based).
"""

