"""Framework-specific exceptions with actionable failure boundaries."""


class LynnRealError(RuntimeError):
    """Base class for expected framework failures."""


class ConfigurationError(LynnRealError):
    """The resolved run configuration violates a public contract."""


class DataContractError(LynnRealError):
    """An index, sample, camera, or tensor violates the data contract."""


class BackendContractError(LynnRealError):
    """A model backend cannot satisfy the requested route."""


class CheckpointError(LynnRealError):
    """A checkpoint transaction or compatibility gate is invalid."""
