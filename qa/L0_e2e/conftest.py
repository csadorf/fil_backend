import os
import socket
from datetime import datetime

import pytest
from hypothesis import Phase, Verbosity, settings
from rapids_triton import Client

settings.register_profile("dev", max_examples=10)
settings.register_profile("ci", max_examples=100)
settings.register_profile(
    "stress",
    max_examples=100000,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    report_multiple_bugs=False,
    verbosity=Verbosity.debug,
)


def pytest_addoption(parser):
    default_repo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_repository"
    )
    parser.addoption("--repo", action="store", default=default_repo_path)


PREDICT_REQUEST_SUCCESS_COUNT = 0


@pytest.fixture(scope="session")
def client():
    """A RAPIDS-Triton client for submitting inference requests"""

    class _Client(Client):
        def predict(self, *args, **kwargs):
            global PREDICT_REQUEST_SUCCESS_COUNT
            ret = super().predict(*args, **kwargs)
            PREDICT_REQUEST_SUCCESS_COUNT += 1
            return ret

    client = _Client()
    client.wait_for_server(120)
    return client


def pytest_sessionfinish(session, exitstatus):
    global PREDICT_REQUEST_SUCCESS_COUNT
    timestamp = datetime.utcnow().isoformat()
    hostname = socket.gethostname()

    print("Writing to test log...")

    log_entry = f"{timestamp} | Hostname: {hostname} | Success Count: {PREDICT_REQUEST_SUCCESS_COUNT}\n"

    with open("/qa/L0_e2e/test_log.txt", "a") as log_file:
        log_file.write(log_entry)

    print(
        f"Final count: The specific test has passed {PREDICT_REQUEST_SUCCESS_COUNT} times."
    )
