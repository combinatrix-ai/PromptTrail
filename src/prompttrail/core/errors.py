from typing import Any


class ParameterValidationError(Exception):
    """Raised when a parameter is invalid."""

    ...


class ConfigurationValidationError(Exception):
    """Raised when a configuration is invalid."""

    ...


class ProviderResponseError(Exception):
    """Raised when a provider returns an error."""

    def __init__(self, message: str, response: Any) -> None:
        """
        Initialize ProviderResponseError.

        Args:
            message (str): The error message.
            response (Any): The response from the provider.
        """
        super().__init__(message)
        self.response = response


class RenderingError(Exception):
    """Raised when a template rendering fails."""

    ...


class HookError(Exception):
    """Raised when a hook fails."""

    ...


class TemplateNotFoundError(Exception):
    """Raised when a template is not found."""

    ...
