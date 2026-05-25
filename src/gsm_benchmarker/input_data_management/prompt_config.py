"""Prompt configuration and formatting for few-shot prompting.

Provides PromptConfig to construct few-shot prompts with customizable formats,
supporting both preset configurations from resources and custom definitions.
Integrates with GSMShotManager to compile few-shot examples.
"""

from dataclasses import dataclass
from pathlib import Path

from gsm_benchmarker.input_data_management.shot_manager import GSMShotManager
from gsm_benchmarker.utils.resources_manager import load_resource_json, load_json_file


@dataclass
class PromptConfig:
    """Configuration for constructing few-shot prompts.

    A PromptConfig defines how to build a prompt by combining an introduction,
    few-shot examples, and a target problem. It supports multiple customization
    dimensions: number of shots, shot format, example separator, and target
    introduction text. It integrates with GSMShotManager to compile examples.

    Attributes:
        n_shots: Number of few-shot examples to include.
        question_format: Format string for questions (must contain '{question}').
        answer_format: Format string for answers (must contain '{solution}').
        intro: Text introducing the task/examples section.
        target_intro: Text introducing the target problem (before the question).
        separator: Separator between intro, examples, target intro, and question.
        shot_intro: Prefix for each shot (e.g., "Example 1:").
        code_type_answer: If True, use code extraction; otherwise textual.
        solutions_name: Name of alternative solutions file (e.g., 'python_functions').

    **Preset configurations** are available in resources/prompt-formats/:
    default, separated-target, code-output-separated-target, etc.
    """

    n_shots: int
    question_format: str
    answer_format: str
    intro: str
    target_intro: str
    separator: str = "\n\n"
    shot_intro: str = ""
    code_type_answer: bool = False
    solutions_name: str | None = None

    def __post_init__(self):
        """Validate format strings and initialize shot manager."""
        if '{question}' not in self.question_format:
            raise ValueError("question_format must contain '{question}' placeholder")

        if '{solution}' not in self.answer_format:
            raise ValueError("answer_format must contain '{solution}' placeholder")

        self.shots = GSMShotManager(self.solutions_name, code=self.code_type_answer)

    @property
    def shot_format(self) -> str:
        """Compiled format string for a single shot (shot_intro + question + answer)."""
        return self.shot_intro + self.question_format + self.answer_format

    def __call__(self, question: str) -> str:
        """Build a complete prompt for the given question.

        Args:
            question: The target problem question.

        Returns:
            Full prompt text with introduction, few-shot examples, and question.
        """

        prompt = self.intro
        prompt += self.separator
        prompt += self.shots.compile(self.shot_format, n_shots=self.n_shots, separator=self.separator)
        prompt += self.separator

        if self.target_intro:
            prompt += self.target_intro
            prompt += self.separator

        prompt += self.question_format.format(question=question)

        return prompt

    @classmethod
    def from_file(cls, file_name: str | Path, **kwargs) -> "PromptConfig":
        """Load prompt configuration from a custom JSON file.

        Args:
            file_name: Path to a JSON file with PromptConfig fields.
            **kwargs: Optional overrides for any fields.

        Returns:
            A PromptConfig instance with settings loaded from file + overrides.
        """
        data_dict = load_json_file(file_name)
        data_dict = data_dict | kwargs  # values from kwargs take precedence
        return cls(**data_dict)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> "PromptConfig":
        """Load prompt configuration from a preset in resources/prompt-formats/.

        Args:
            preset_name: Name of the preset (e.g., 'default', 'separated-target').
            **kwargs: Optional overrides for any fields.

        Returns:
            A PromptConfig instance with settings from the preset + overrides.

        Raises:
            ValueError: If the preset_name does not exist in resources.
        """
        try:
            data_dict = load_resource_json(f"prompt-formats/{preset_name}.json")
        except FileNotFoundError:
            raise ValueError(f"Preset '{preset_name}' is not known")
        data_dict = data_dict | kwargs
        return cls(**data_dict)

    @classmethod
    def default(cls, **kwargs) -> "PromptConfig":
        """Load the default preset.

        Args:
            **kwargs: Optional overrides for any fields.

        Returns:
            A PromptConfig instance using the 'default' preset + overrides.
        """
        return cls.from_preset("default", **kwargs)


if __name__ == '__main__':
    pc_default = PromptConfig.default(n_shots=3)
    print("DEFAULT PROMPT\n")
    print(pc_default("<Question here>"))

    print("\n" + 20 * "=" + "\n")

    pc_code = PromptConfig.from_preset('code-output-separated-target', n_shots=2)
    print('CODE PROMPT\n')
    print(pc_code("<Question here>"))
