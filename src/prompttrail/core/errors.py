from typing import Any


class ParameterValidationError(Exception):
    """Raised when a parameter validation fails."""


class ConfigurationValidationError(Exception):
    """Raised when a configuration validation fails."""


class ProviderResponseError(Exception):
    """Raised when a provider returns an error response."""

    def __init__(self, message: str, response: Any) -> None:
        """Initialize ProviderResponseError.

        Args:
            message: The error message
            response: The raw response from the provider
        """
        super().__init__(message)
        self.response = response


class RenderingError(Exception):
    """Raised when template rendering fails."""


class HookError(Exception):
    """Raised when a hook execution fails."""


class TemplateNotFoundError(Exception):
    """Raised when a referenced template cannot be found."""
