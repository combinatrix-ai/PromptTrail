import pytest
import logging
from examples.dogfooding.dogfooding_tools import disable_noisy_loggers

def pytest_configure(config):
    disable_noisy_loggers()