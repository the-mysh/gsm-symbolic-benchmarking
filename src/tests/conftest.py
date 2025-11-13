import pytest

from gsm_benchmarker.shot_manager import GSMShotManager


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
def mock_shot_manager(mocker):
    mocker.patch(
        "gsm_benchmarker.shot_manager.load_resource_json",
        return_value=MOCK_8SHOT_DATA
    )

    # 2. Instantiate and return the object that uses the patched function
    manager = GSMShotManager()
    return manager
