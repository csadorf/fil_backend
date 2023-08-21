import os

from hypothesis import Phase, Verbosity, settings

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
