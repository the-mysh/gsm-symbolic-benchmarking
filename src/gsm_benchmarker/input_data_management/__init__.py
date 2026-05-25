"""Input data management for GSM-Symbolic benchmarking.

This subpackage handles dataset loading and prompt configuration:

* `GSMSymbolicDataset`: Load GSM-Symbolic dataset variants with instance-based
  filtering for reproducibility.
* `PromptConfig`: Configure prompt formats with few-shot examples, supporting
  both presets and custom formats.
* `GSMShotManager`: Manage and compile few-shot examples with format flexibility.

**Typical usage:**

    # Load dataset variant
    dset = GSMSymbolicDataset(variant=GSMSymbolicDataset.Variant.main)
    evaluation_sets = dset.create_evaluation_sets()

    # Configure prompt with preset
    prompt_config = PromptConfig.from_preset('default', n_shots=3)

    # Create prompt for an example
    prompt = prompt_config(question="What is 5 + 3?")
"""

