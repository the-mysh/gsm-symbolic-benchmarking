"""Results analyser package.

This package provides helpers and analysers for processing and visualising
results produced by the GSM benchmarking pipeline. Exposes top-level analyser
classes for single-model results, collections of models, and multi-variant
comparisons.
"""

from gsm_benchmarker.results_analyser.model import ModelResultsAnalyser
from gsm_benchmarker.results_analyser.multi_model import MultiModelResultsAnalyser
from gsm_benchmarker.results_analyser.multi_variant_multi_model import MultiVariantMultiModelResultsAnalyser
