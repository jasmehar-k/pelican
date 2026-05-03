"""
Classic factor library — 8 well-documented signals spanning all major factor families.

Importing this package registers all signals into the backtest engine's registry.
Price-based: MOM_1_12, REVERSAL_1M, LOW_VOL
Fundamental: SIZE, VALUE_PE, VALUE_PB, QUALITY_ROE, QUALITY_LEVERAGE
Alternative data: EDGAR_SENTIMENT, NEWS_SENTIMENT
"""

# Importing these modules registers signals as a side effect of the @register decorator.
from pelican.factors import edgar_sentiment, low_vol, momentum, news_sentiment, quality, size, value  # noqa: F401
from pelican.factors.correlation import build_factor_correlation_matrix, plot_correlation_heatmap

ALL_FACTORS = [
    "MOM_1_12",
    "REVERSAL_1M",
    "LOW_VOL",
    "SIZE",
    "VALUE_PE",
    "VALUE_PB",
    "QUALITY_ROE",
    "QUALITY_LEVERAGE",
    "EDGAR_SENTIMENT",
    "NEWS_SENTIMENT",
]

__all__ = [
    "ALL_FACTORS",
    "build_factor_correlation_matrix",
    "plot_correlation_heatmap",
]
