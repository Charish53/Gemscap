"""Custom exceptions for market analytics system."""


class MarketDataError(Exception):
    """Base exception for market data related errors."""
    pass


class ConnectorError(MarketDataError):
    """Exception raised when connector fails."""
    pass


class StorageError(MarketDataError):
    """Exception raised when storage operation fails."""
    pass


class AnalyticsError(Exception):
    """Base exception for analytics computation errors."""
    pass


class ResamplingError(AnalyticsError):
    """Exception raised when resampling fails."""
    pass


class AlertError(Exception):
    """Exception raised when alert system fails."""
    pass

