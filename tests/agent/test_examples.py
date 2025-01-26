# This script run tests using examples as integrated tests.

import os
import sys

# Import examples as module
path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
from examples.agent import fermi_problem  # noqa: E402


def test_fermi_problem():
    # In the example, mocked model is loaded in pytest environment.
    _ = fermi_problem.runner.run(max_messages=10)
    # TODO: Can be done without mocking? But it may make unintended loop or something.
