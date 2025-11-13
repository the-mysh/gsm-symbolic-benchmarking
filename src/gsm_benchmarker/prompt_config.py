from dataclasses import dataclass
from pathlib import Path

from gsm_benchmarker.shot_manager import GSMShotManager
from gsm_benchmarker.utils.resources_manager import load_resource_json, load_json_file


@dataclass
class PromptConfig:
    n_shots: int
    question_format: str
    answer_format: str
    intro: str
    target_intro: str
    separator: str = "\n\n"

    def __post_init__(self):
        if '{question}' not in self.question_format:
            raise ValueError("question_format must contain '{question}' placeholder")

        if '{solution}' not in self.answer_format:
            raise ValueError("answer_format must contain '{solution}' placeholder")

    @property
    def shot_format(self) -> str:
        return self.question_format + self.answer_format

    def __call__(self, question: str, shots: GSMShotManager) -> str:

        prompt = self.intro
        prompt += self.separator
        prompt += shots.compile(self.shot_format, n_shots=self.n_shots, separator=self.separator)
        prompt += self.separator

        if self.target_intro:
            prompt += self.target_intro
            prompt += self.separator

        prompt += self.question_format.format(question=question)

        return prompt

    @classmethod
    def from_file(cls, file_name: str | Path, **kwargs) -> "PromptConfig":
        data_dict = load_json_file(file_name)
        data_dict = data_dict | kwargs  # values from kwargs take precedence
        return cls(**data_dict)

    @classmethod
    def from_preset(cls, preset_name: str, **kwargs) -> "PromptConfig":
        data_dict = load_resource_json(f"prompt-formats/{preset_name}.json")
        data_dict = data_dict | kwargs
        return cls(**data_dict)

    @classmethod
    def default(cls, **kwargs) -> "PromptConfig":
        return cls.from_preset("default", **kwargs)
