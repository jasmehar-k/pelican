"""Tests for Stage 8: EDGAR sentiment agent.

All network calls and LLM calls are mocked — no internet access required.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pelican.data.edgar import (
    _compute_tone_deltas,
    _strip_html,
    extract_mda,
    fetch_filing_metadata,
    get_cik,
    score_tone,
    seed_edgar_sentiment,
)
from pelican.data.store import DataStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp."},
}

_SAMPLE_SUBMISSIONS = {
    "cik": "0000320193",
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-Q", "8-K"],
            "filingDate": ["2023-11-03", "2023-08-04", "2023-05-05", "2023-11-01"],
            "reportDate": ["2023-09-30", "2023-07-01", "2023-04-01", ""],
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-23-000055",
                "0000320193-23-000099",
            ],
            "primaryDocument": [
                "aapl-20230930.htm",
                "aapl-20230701.htm",
                "aapl-20230401.htm",
                "8k.htm",
            ],
        }
    },
}

_SAMPLE_HTML = """
<html><head><title>10-K</title></head>
<body>
<p>Introduction text here.</p>
<p><b>Item 7. Management's Discussion and Analysis</b></p>
<p>Revenue increased 8% driven by strong iPhone sales. Operating margins improved.
Outlook remains positive for the next fiscal year.</p>
<p><b>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</b></p>
<p>We are exposed to market risk in the ordinary course of business.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class TestStripHTML:
    def test_removes_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result
        assert ">" not in result

    def test_skips_script_content(self):
        html = "<script>alert('bad')</script><p>Good content</p>"
        result = _strip_html(html)
        assert "bad" not in result
        assert "Good content" in result

    def test_skips_style_content(self):
        html = "<style>.foo{color:red}</style><p>Text</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "Text" in result

    def test_handles_html_entities(self):
        html = "<p>AT&amp;T earns &gt; $100B</p>"
        result = _strip_html(html)
        assert "AT&T" in result
        assert "$100B" in result


# ---------------------------------------------------------------------------
# MD&A extraction
# ---------------------------------------------------------------------------

class TestExtractMda:
    def test_finds_item_7_section(self):
        result = extract_mda(_SAMPLE_HTML)
        assert "Management" in result
        assert "Revenue" in result

    def test_stops_at_item_7a(self):
        result = extract_mda(_SAMPLE_HTML)
        assert "Market Risk" not in result

    def test_returns_empty_when_no_item_7(self):
        # No fallback to raw text — avoids sending XBRL garbage to scorer as a false 0.0
        html = "<html><body><p>No standard structure here. Just text.</p></body></html>"
        result = extract_mda(html)
        assert result == ""

    def test_output_never_exceeds_max_chars(self):
        long_html = "<p>Item 7. MD&A</p>" + "<p>" + "x" * 10000 + "</p>"
        result = extract_mda(long_html)
        assert len(result) <= 4_000

    def test_strips_html_in_mda(self):
        prose = " Revenue and operating income both grew substantially year over year." * 4
        html = f"<p>Item 7.</p><p><b>Strong growth</b> in <i>all segments</i>.{prose}</p><p>Item 8.</p>"
        result = extract_mda(html)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "Strong growth" in result

    def test_returns_empty_for_pure_xbrl_document(self):
        xbrl = "<?xml version='1.0' encoding='ASCII'?>\n<xbrl xmlns='http://xbrl.org/2005'><context/></xbrl>"
        result = extract_mda(xbrl)
        assert result == ""

    def test_ixbrl_namespace_tags_stripped(self):
        prose = " Net revenue increased driven by strong product demand across all regions." * 3
        html = (
            f"<html><body><p>Item 7.</p>"
            f"<p><ix:nonfraction>1234</ix:nonfraction> Revenue grew.{prose}</p>"
            f"<p>Item 8.</p></body></html>"
        )
        result = extract_mda(html)
        assert "1234" in result
        assert "ix:nonfraction" not in result


# ---------------------------------------------------------------------------
# EDGAR API — CIK lookup
# ---------------------------------------------------------------------------

class TestGetCik:
    def test_returns_zero_padded_cik(self):
        with patch("pelican.data.edgar._load_cik_map", return_value={"AAPL": "0000320193"}):
            result = get_cik("AAPL", user_agent="test test@test.com")
        assert result == "0000320193"

    def test_case_insensitive(self):
        with patch("pelican.data.edgar._load_cik_map", return_value={"AAPL": "0000320193"}):
            result = get_cik("aapl", user_agent="test test@test.com")
        assert result == "0000320193"

    def test_returns_none_for_unknown_ticker(self):
        with patch("pelican.data.edgar._load_cik_map", return_value={"AAPL": "0000320193"}):
            result = get_cik("ZZZZ", user_agent="test test@test.com")
        assert result is None


# ---------------------------------------------------------------------------
# EDGAR API — filing metadata
# ---------------------------------------------------------------------------

class TestFetchFilingMetadata:
    def _mock_get(self, *args, **kwargs):
        m = MagicMock()
        m.json.return_value = _SAMPLE_SUBMISSIONS
        m.raise_for_status.return_value = None
        return m

    def test_returns_10k_entries(self):
        with patch("pelican.data.edgar._edgar_get", side_effect=self._mock_get), \
             patch("time.sleep"):
            results = fetch_filing_metadata(
                "0000320193", "10-K", "test@test.com",
                after=date(2020, 1, 1), before=date(2024, 1, 1),
            )
        assert len(results) == 1
        assert results[0]["filing_type"] == "10-K"
        assert results[0]["filing_date"] == date(2023, 11, 3)
        assert results[0]["period_end"] == date(2023, 9, 30)

    def test_returns_10q_entries(self):
        with patch("pelican.data.edgar._edgar_get", side_effect=self._mock_get), \
             patch("time.sleep"):
            results = fetch_filing_metadata(
                "0000320193", "10-Q", "test@test.com",
                after=date(2020, 1, 1), before=date(2024, 1, 1),
            )
        assert len(results) == 2
        assert all(r["filing_type"] == "10-Q" for r in results)

    def test_excludes_8k(self):
        with patch("pelican.data.edgar._edgar_get", side_effect=self._mock_get), \
             patch("time.sleep"):
            results = fetch_filing_metadata(
                "0000320193", "8-K", "test@test.com",
            )
        assert len(results) == 1  # only the 8-K entry, not 10-K or 10-Q

    def test_respects_limit(self):
        with patch("pelican.data.edgar._edgar_get", side_effect=self._mock_get), \
             patch("time.sleep"):
            results = fetch_filing_metadata(
                "0000320193", "10-Q", "test@test.com", limit=1,
            )
        assert len(results) == 1

    def test_filters_by_date_range(self):
        with patch("pelican.data.edgar._edgar_get", side_effect=self._mock_get), \
             patch("time.sleep"):
            results = fetch_filing_metadata(
                "0000320193", "10-K", "test@test.com",
                after=date(2024, 1, 1),  # after all available filings
            )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tone scoring
# ---------------------------------------------------------------------------

class TestScoreTone:
    def _make_llm_mock(self, content: str):
        mock_resp = MagicMock()
        mock_resp.content = content
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_resp
        return mock_llm

    def test_parses_valid_json(self):
        with patch("pelican.data.edgar._get_llm",
                   return_value=self._make_llm_mock('{"tone_score": 0.4}')):
            result = score_tone("Revenue increased significantly this year.")
        assert result == pytest.approx(0.4)

    def test_clamps_above_1(self):
        with patch("pelican.data.edgar._get_llm",
                   return_value=self._make_llm_mock('{"tone_score": 2.5}')):
            result = score_tone("Very positive outlook.")
        assert result == pytest.approx(1.0)

    def test_clamps_below_minus1(self):
        with patch("pelican.data.edgar._get_llm",
                   return_value=self._make_llm_mock('{"tone_score": -3.0}')):
            result = score_tone("Severe warnings and risks.")
        assert result == pytest.approx(-1.0)

    def test_returns_none_for_invalid_json(self):
        with patch("pelican.data.edgar._get_llm",
                   return_value=self._make_llm_mock("I cannot determine the tone.")):
            result = score_tone("Some text.")
        assert result is None

    def test_returns_none_for_empty_text(self):
        result = score_tone("")
        assert result is None

    def test_returns_none_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM error")
        with patch("pelican.data.edgar._get_llm", return_value=mock_llm):
            result = score_tone("Some filing text.")
        assert result is None

    def test_negative_score_parsed(self):
        with patch("pelican.data.edgar._get_llm",
                   return_value=self._make_llm_mock('{"tone_score": -0.6}')):
            result = score_tone("Revenues declined and risks increased.")
        assert result == pytest.approx(-0.6)


# ---------------------------------------------------------------------------
# Tone delta computation
# ---------------------------------------------------------------------------

class TestComputeToneDeltas:
    def test_computes_delta_for_prior_year(self):
        records = [
            {
                "ticker": "AAPL", "filing_type": "10-K",
                "period_end": date(2022, 9, 30),
                "tone_score": 0.3, "filing_date": date(2022, 11, 1),
            },
            {
                "ticker": "AAPL", "filing_type": "10-K",
                "period_end": date(2023, 9, 30),
                "tone_score": 0.7, "filing_date": date(2023, 11, 1),
            },
        ]
        result = _compute_tone_deltas(records)
        by_period = {r["period_end"]: r for r in result}
        assert by_period[date(2023, 9, 30)]["tone_delta"] == pytest.approx(0.4)
        assert by_period[date(2022, 9, 30)]["tone_delta"] is None

    def test_null_when_no_prior_year(self):
        records = [
            {
                "ticker": "AAPL", "filing_type": "10-K",
                "period_end": date(2023, 9, 30),
                "tone_score": 0.5, "filing_date": date(2023, 11, 1),
            },
        ]
        result = _compute_tone_deltas(records)
        assert result[0]["tone_delta"] is None

    def test_tolerates_45_day_slack(self):
        records = [
            {
                "ticker": "AAPL", "filing_type": "10-Q",
                "period_end": date(2022, 6, 25),  # 40 days before 7/5
                "tone_score": 0.2, "filing_date": date(2022, 8, 1),
            },
            {
                "ticker": "AAPL", "filing_type": "10-Q",
                "period_end": date(2023, 7, 5),
                "tone_score": 0.6, "filing_date": date(2023, 8, 4),
            },
        ]
        result = _compute_tone_deltas(records)
        by_period = {r["period_end"]: r for r in result}
        assert by_period[date(2023, 7, 5)]["tone_delta"] == pytest.approx(0.4)

    def test_handles_empty_records(self):
        result = _compute_tone_deltas([])
        assert result == []

    def test_skips_null_tone_scores(self):
        records = [
            {
                "ticker": "AAPL", "filing_type": "10-K",
                "period_end": date(2022, 9, 30),
                "tone_score": None, "filing_date": date(2022, 11, 1),
            },
            {
                "ticker": "AAPL", "filing_type": "10-K",
                "period_end": date(2023, 9, 30),
                "tone_score": 0.5, "filing_date": date(2023, 11, 1),
            },
        ]
        result = _compute_tone_deltas(records)
        by_period = {r["period_end"]: r for r in result}
        # prior year had None tone_score so delta should be None
        assert by_period[date(2023, 9, 30)]["tone_delta"] is None


# ---------------------------------------------------------------------------
# DataStore schema
# ---------------------------------------------------------------------------

class TestEdgarSentimentTable:
    def test_schema_creates_edgar_table(self):
        store = DataStore(":memory:")
        store.init_schema()
        tables = store.query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        )["table_name"].to_list()
        assert "edgar_sentiment" in tables

    def test_store_edgar_scores_round_trip(self):
        store = DataStore(":memory:")
        store.init_schema()
        df = pl.DataFrame({
            "ticker": ["AAPL"],
            "filing_date": [date(2023, 11, 3)],
            "period_end": [date(2023, 9, 30)],
            "filing_type": ["10-K"],
            "tone_score": [0.5],
            "tone_delta": [0.2],
            "model": ["test-model"],
        })
        n = store.store_edgar_scores(df)
        assert n == 1
        result = store.query("SELECT * FROM edgar_sentiment")
        assert len(result) == 1
        assert result["ticker"][0] == "AAPL"
        assert result["tone_delta"][0] == pytest.approx(0.2)

    def test_get_edgar_coverage_empty(self):
        store = DataStore(":memory:")
        store.init_schema()
        result = store.get_edgar_coverage()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Signal spec
# ---------------------------------------------------------------------------

class TestEdgarSignalSpec:
    def test_requires_edgar_default_false(self):
        from pelican.backtest.signals import SignalSpec
        spec = SignalSpec(name="TEST", description="test")
        assert spec.requires_edgar is False

    def test_edgar_sentiment_requires_edgar(self):
        import pelican.factors.edgar_sentiment  # noqa: F401 — triggers registration
        from pelican.backtest.signals import get_signal
        sig = get_signal("EDGAR_SENTIMENT")
        assert sig.spec.requires_edgar is True
        assert "tone_delta" in sig.spec.edgar_data_deps

    def test_edgar_sentiment_signal_fn(self):
        import pelican.factors.edgar_sentiment  # noqa: F401
        from pelican.backtest.signals import get_signal
        sig = get_signal("EDGAR_SENTIMENT")
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "tone_delta": [0.3, -0.1],
        })
        result = sig.fn(df)
        assert len(result) == 2
        assert result.name == "EDGAR_SENTIMENT"
        assert result[0] == pytest.approx(0.3)
        assert result[1] == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Engine PIT join
# ---------------------------------------------------------------------------

class TestEdgarPitJoin:
    def test_engine_joins_edgar_panel(self):
        """Engine attaches tone_delta to cross-section when requires_edgar=True."""
        from datetime import date
        from unittest.mock import patch

        from pelican.backtest.engine import BacktestConfig, run_backtest
        from pelican.backtest.signals import SignalDef, SignalSpec, _REGISTRY

        # Register a temporary signal that reads tone_delta.
        def _test_signal(cs: pl.DataFrame) -> pl.Series:
            return cs["tone_delta"].alias("_TEST_EDGAR")

        spec = SignalSpec(
            name="_test_edgar_join",
            description="test",
            requires_edgar=True,
        )
        _REGISTRY["_test_edgar_join"] = SignalDef(spec=spec, fn=_test_signal)

        try:
            store = DataStore(":memory:")
            store.init_schema()

            prices = pl.DataFrame({
                "ticker": ["A", "A", "B", "B"],
                "date": [date(2023, 1, 3), date(2023, 2, 1),
                         date(2023, 1, 3), date(2023, 2, 1)],
                "open": [100.0, 101.0, 50.0, 51.0],
                "high": [102.0, 103.0, 52.0, 53.0],
                "low":  [99.0,  100.0, 49.0, 50.0],
                "close": [101.0, 102.0, 51.0, 52.0],
                "volume": [1000, 1100, 500, 550],
                "log_return_1d": [0.01, 0.01, 0.01, 0.01],
                "forward_return_21d": [0.02, 0.02, -0.01, -0.01],
            })
            edgar = pl.DataFrame({
                "ticker": ["A", "B"],
                "filing_date": [date(2023, 1, 2), date(2023, 1, 2)],
                "period_end": [date(2022, 12, 31), date(2022, 12, 31)],
                "filing_type": ["10-K", "10-K"],
                "tone_score": [0.4, -0.2],
                "tone_delta": [0.1, -0.3],
                "model": ["test", "test"],
            })
            universe = pl.DataFrame({
                "ticker": ["A", "B"],
                "entry_date": [date(2022, 1, 1), date(2022, 1, 1)],
                "exit_date": [None, None],
                "company": ["Alpha", "Beta"],
            })
            store.write(prices, "prices")
            store.write(universe, "sp500_universe")
            store.store_edgar_scores(edgar)

            config = BacktestConfig(
                start=date(2023, 1, 1),
                end=date(2023, 3, 1),
                min_universe_size=2,
                min_score_coverage=0.3,
            )

            # Patch universe query to return our 2 test tickers.
            with patch(
                "pelican.backtest.engine.get_point_in_time_universe",
                return_value=["A", "B"],
            ):
                result = run_backtest("_test_edgar_join", config, store)

            assert result.n_periods >= 1
        finally:
            _REGISTRY.pop("_test_edgar_join", None)
