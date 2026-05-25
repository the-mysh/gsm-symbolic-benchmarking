import pytest

from gsm_benchmarker.input_data_management.shot_manager import GSMShotManager


MOCK_8SHOT_DATA = {
    "comment": "Mock 8 shots for testing.",
    "samples": [
        {
            "question": "Q1?",
            "solution": "A1.",
            "result": "11"
        },
        {
            "question": "Q2?",
            "solution": "A2.",
            "result": "25"
        },
        {
            "question": "Q3?",
            "solution": "A3.",
            "result": "39"
        },
    ]
}


@pytest.fixture
def shot_manager():
    return GSMShotManager()


@pytest.fixture
def mock_shot_manager(monkeypatch):
    monkeypatch.setattr(
        "gsm_benchmarker.input_data_management.shot_manager.load_resource_json",
        lambda *args, **kwargs: MOCK_8SHOT_DATA,
    )

    manager = GSMShotManager()
    return manager
