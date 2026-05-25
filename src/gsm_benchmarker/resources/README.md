# Resources

This directory contains configuration files and presets used by the GSM-Symbolic benchmarking framework.

## Directory Contents

### Configuration Files (Root Level)

#### `machines_config.json`
Machine and cluster configuration mapping.

- **Purpose**: Define compute resources available on each machine in a computing cluster.
- **Structure**:
  - `machine_types`: Hardware specifications (RAM, VRAM, GPU count, bfloat16 support) for each machine type
  - `machines`: Mapping from machine names to their types
- **Usage**: Loaded by `BenchmarkConfig.for_machine()` to automatically configure memory allocation and GPU settings based on the machine name
- **Example machines**: `lima`, `douro`, `cavado`, `guadiana` (target systems for the INESC-ID cluster)

#### `original-models-config.json`
Model catalog with metadata for all available models.

- **Purpose**: Define model names, families, sizes, and provider information.
- **Structure**: List of model objects with fields:
  - `name`: HuggingFace model ID or API model name
  - `family`: Model family (e.g., "gemma", "llama", "phi", "gpt")
  - `size`: Parameter count in billions (or scientific notation)
  - `instruction_tuned`: Whether the model is instruction-fine-tuned
  - `api_type`: `null` for local HF models, or `"openai"`, `"anthropic"`, `"google_genai"` for API models
  - `extra_kwargs`: Provider-specific kwargs for model loading
- **Usage**: Loaded by `ModelsConfig` to provide model lookup and filtering. Reference the model name when running benchmarks.

#### `standard-8-shots.json`
Standard few-shot examples for chain-of-thought prompting.

- **Purpose**: Provide a common set of 8 mathematical examples used across benchmarks.
- **Structure**: List of 8 examples, each with:
  - `question`: The problem statement
  - `solution`: Chain-of-thought reasoning
  - `result`: The numerical answer
- **Source**: Reproduced from the EleutherAI LM evaluation harness (GSM8K task)
- **Usage**: Loaded by `GSMShotManager` as the default shot set. Can be overridden with alternative solution formats from the `solutions_8shot/` directory.

### Prompt Format Presets

#### `prompt-formats/` Directory
Alternative prompt template configurations for different reasoning styles and output formats.

Each JSON file defines fields for the `PromptConfig` dataclass:
- `n_shots`: Number of few-shot examples to include
- `question_format`: Format string for the problem (must contain `{question}` placeholder)
- `answer_format`: Format string for the solution (must contain `{solution}` placeholder)
- `intro`: Introductory text for the task
- `target_intro`: Optional text introducing the target problem
- `separator`: String joining prompt sections (default: "\n\n")
- `shot_intro`: Optional prefix for each shot (e.g., "Example 1:")
- `code_type_answer`: Whether the answer format is Python code
- `solutions_name`: Name of alternative solution format to use (if any)

**Available Presets:**

- **`default.json`**: Standard chain-of-thought format with "Let's think step by step" trigger
- **`separated-target.json`**: Separates example section from target problem with explicit header
- **`separated-target-thinking-trigger.json`**: Like separated-target but with thinking trigger phrases
- **`visual-separated-target.json`**: Uses visual separators (e.g., dashes, lines) between sections
- **`nonformalised.json`**: Informal, conversational reasoning style
- **`formalised.json`**: Formal, structured mathematical notation style
- **`no-step-by-step.json`**: Direct answer format without intermediate steps
- **`code-output-*.json`** (5 variants): Code-based answers with different output styles:
  - `code-output-separated-target.json`: Code format with separated target
  - `code-output-no-sep.json`: Code format without separators
  - `code-output-no-sep-short.json`: Compact code format
  - `code-output-no-sep-no-variables.json`: Code without intermediate variables
  - (Additional variants for specific experimental configurations)

**Usage**: Load with `PromptConfig.from_preset(preset_name, **overrides)` or reference in model config `solutions_name` field.

### Alternative Solution Sets

#### `solutions_8shot/` Directory
Alternative Python code implementations of the standard 8-shot examples.

Each Python file defines a `SOLUTIONS` list with 8 functions implementing the solutions to the standard examples.

**Available Variants:**

- **`python_functions.py`**: Full, well-commented Python functions with variable names and step-by-step logic
- **`python_functions_short.py`**: Compact Python implementations with minimal comments
- **`python_functions_no_variables.py`**: Inline implementations without intermediate variable assignments
- **`formalised.py`**: Solutions using formal mathematical notation and libraries
- **`nonformalised.py`**: Solutions with informal, narrative-style comments

**Usage**: Reference the file stem (without `.py`) in:
- Model config: `"extra_kwargs": {"solutions_name": "python_functions"}`
- Prompt config: `"solutions_name": "python_functions"`

Loaded by `GSMShotManager(solutions_name="python_functions", code=True)` to provide code-based few-shot examples.

---

## Common Usage Patterns

### Load a preset prompt format:
```python
from gsm_benchmarker.input_data_management.prompt_config import PromptConfig

# Use the default preset
pc = PromptConfig.default(n_shots=8)

# Use a custom preset with overrides
pc = PromptConfig.from_preset('separated-target', n_shots=3, target_intro="Solve this:")
```

### Load configuration for a specific machine:
```python
from gsm_benchmarker.benchmark.benchmark_config import BenchmarkConfig

# Auto-detect machine from hostname and load its config
config = BenchmarkConfig.for_machine("douro", ram_margin=4, vram_margin=2)
```

### Look up a model:
```python
from gsm_benchmarker.model_wrappers.models_config_parser import ModelsConfig

models = ModelsConfig()
model = models["google/gemma-7b-it"]
print(f"Model: {model.name}, Size: {model.size}B, API: {model.api_type}")
```

### Use alternative code-based few-shots:
```python
from gsm_benchmarker.input_data_management.prompt_config import PromptConfig

# Create a code-based prompt with compact Python functions
pc = PromptConfig.from_preset('code-output-separated-target', 
                              n_shots=3, 
                              solutions_name='python_functions_short')
```

