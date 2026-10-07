class JuryError(Exception):
    """Base exception for expected mock-jury failures."""


class CaseFileError(JuryError):
    """The case file could not be loaded."""


class ProviderError(JuryError):
    """The configured LLM provider could not return a usable response."""


class InvalidJurorResponse(JuryError):
    """A juror response did not match the required schema."""
