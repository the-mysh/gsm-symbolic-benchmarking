"""Stopping criteria for language model generation.

Provides utilities for controlling when a model should stop generating tokens
during inference, allowing early exit based on custom string patterns.
"""

import torch
from transformers import StoppingCriteria


class StopOnStringCriteria(StoppingCriteria):
    """Stop text generation when a specified string appears in the output.

    This stopping criterion decodes the last few generated tokens and checks
    for the presence of any stop string. It enables efficient early stopping
    without decoding the entire sequence at each generation step.
    """

    def __init__(self, stop_strings: list[str], tokenizer):
        """Initialize the stopping criterion.

        Args:
            stop_strings: List of strings that, when detected in the output,
                will trigger generation to stop.
            tokenizer: Tokenizer to use for decoding generated tokens.
        """
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if any stop string appears in the recently generated tokens.

        Args:
            input_ids: Generated token IDs so far.
            scores: Logits from the latest generation step.
            **kwargs: Additional arguments (ignored, for HuggingFace compatibility).

        Returns:
            True if any stop string is found in the last ~10 tokens, False otherwise.
        """
        # Decode only the last few tokens to check for the stop string.
        # Decoding the whole sequence every step is too slow.
        tail_tokens = input_ids[0][-10:]
        generated_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)

        for stop_str in self.stop_strings:
            if stop_str in generated_text:
                return True
        return False
