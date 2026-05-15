"""Tests for the theme discoverer node and search_arxiv_recent."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pelican.agents.theme_discoverer import (
    _FALLBACK_THEMES,
    _build_user_message,
    _parse_themes,
    _make_theme_discoverer_node,
    discover_theme,
)
from pelican.agents.tools.search import search_arxiv_recent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atom_feed(entries: list[dict]) -> str:
    """Build a minimal arXiv Atom feed XML string."""
    items = ""
    for e in entries:
        items += f"""
        <entry xmlns="http://www.w3.org/2005/Atom">
          <id>http://arxiv.org/abs/{e['id']}v1</id>
          <title>{e.get('title', 'Test Paper')}</title>
          <summary>{e.get('abstract', 'Test abstract.')}</summary>
          <author><name>{e.get('author', 'Author A')}</name></author>
        </entry>"""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">{items}</feed>"""


def _mock_response(feed_xml: str, status: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = feed_xml
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# TestSearchArxivRecent
# ---------------------------------------------------------------------------

class TestSearchArxivRecent:
    def test_passes_submitted_date_sort(self):
        feed = _atom_feed([{"id": "2501.00001", "title": "Factor Paper"}])
        captured = {}

        def fake_get(url, params, timeout):
            captured.update(params)
            return _mock_response(feed)

        with patch("httpx.get", side_effect=fake_get):
            results = search_arxiv_recent(n=5)

        assert captured.get("sortBy") == "submittedDate"
        assert captured.get("max_results") == 5
        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2501.00001"

    def test_empty_feed_returns_empty_list(self):
        feed = _atom_feed([])
        with patch("httpx.get", return_value=_mock_response(feed)):
            results = search_arxiv_recent()
        assert results == []

    def test_network_error_returns_empty_list(self):
        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            results = search_arxiv_recent()
        assert results == []

    def test_uses_only_qfin_categories_in_query(self):
        feed = _atom_feed([])
        captured = {}

        def fake_get(url, params, timeout):
            captured.update(params)
            return _mock_response(feed)

        with patch("httpx.get", side_effect=fake_get):
            search_arxiv_recent()

        query = captured.get("search_query", "")
        assert "q-fin" in query
        # No keyword terms — only category filter
        assert "abs:" not in query
        assert "ti:" not in query


# ---------------------------------------------------------------------------
# TestParseThemes
# ---------------------------------------------------------------------------

class TestParseThemes:
    def test_parses_all_three(self):
        text = (
            "THEME_1: Momentum signal using close_252d.\n"
            "THEME_2: Value signal using pb_ratio.\n"
            "THEME_3: Low-vol using vol_63d.\n"
        )
        themes = _parse_themes(text)
        assert len(themes) == 3
        assert "Momentum" in themes[0]
        assert "Value" in themes[1]
        assert "Low-vol" in themes[2]

    def test_partial_response_returns_partial_list(self):
        text = "THEME_1: Momentum using close_252d.\n"
        themes = _parse_themes(text)
        assert len(themes) == 1

    def test_empty_response_returns_empty(self):
        assert _parse_themes("") == []

    def test_case_insensitive(self):
        text = "theme_1: Reversal signal.\ntheme_2: Quality signal.\ntheme_3: Value signal.\n"
        themes = _parse_themes(text)
        assert len(themes) == 3

    def test_short_stubs_rejected(self):
        text = "THEME_1: ok\nTHEME_2: Proper signal using vol_21d for low-vol premium.\n"
        themes = _parse_themes(text)
        # "ok" is <= 10 chars, rejected
        assert len(themes) == 1
        assert "Proper" in themes[0]


# ---------------------------------------------------------------------------
# TestDiscoverTheme
# ---------------------------------------------------------------------------

_SAMPLE_PAPERS = [
    {
        "title": "Momentum and Drift in Equity Markets",
        "authors": ["Author A"],
        "abstract": "We study the momentum anomaly in equity returns.",
        "arxiv_id": "2501.00001",
        "url": "https://arxiv.org/abs/2501.00001",
    },
]

_VALID_LLM_RESPONSE = (
    "THEME_1: Exploit momentum persistence using close_252d / close_21d ratio "
    "as documented in paper 2501.00001.\n"
    "THEME_2: Low volatility premium using negative vol_63d as per 2501.00002.\n"
    "THEME_3: Value reversion using inverse pb_ratio from 2501.00003.\n"
)


class TestDiscoverTheme:
    def _mock_llm(self, content: str):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=content)
        return llm

    def test_returns_first_parsed_theme(self):
        with (
            patch("pelican.agents.theme_discoverer.search_arxiv_recent", return_value=_SAMPLE_PAPERS),
            patch("pelican.agents.theme_discoverer.list_signals", return_value=[]),
            patch("pelican.agents.theme_discoverer._get_llm", return_value=self._mock_llm(_VALID_LLM_RESPONSE)),
        ):
            result = discover_theme()
        assert "momentum" in result.lower() or "close_252d" in result.lower()

    def test_fallback_on_empty_papers(self):
        with patch("pelican.agents.theme_discoverer.search_arxiv_recent", return_value=[]):
            result = discover_theme()
        assert result == _FALLBACK_THEMES[0]

    def test_fallback_on_arxiv_exception(self):
        with patch(
            "pelican.agents.theme_discoverer.search_arxiv_recent",
            side_effect=RuntimeError("network error"),
        ):
            result = discover_theme()
        assert result == _FALLBACK_THEMES[0]

    def test_fallback_on_llm_failure(self):
        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = RuntimeError("LLM down")
        with (
            patch("pelican.agents.theme_discoverer.search_arxiv_recent", return_value=_SAMPLE_PAPERS),
            patch("pelican.agents.theme_discoverer.list_signals", return_value=[]),
            patch("pelican.agents.theme_discoverer._get_llm", return_value=bad_llm),
        ):
            result = discover_theme()
        assert result == _FALLBACK_THEMES[0]

    def test_fallback_on_unparseable_response(self):
        with (
            patch("pelican.agents.theme_discoverer.search_arxiv_recent", return_value=_SAMPLE_PAPERS),
            patch("pelican.agents.theme_discoverer.list_signals", return_value=[]),
            patch("pelican.agents.theme_discoverer._get_llm", return_value=self._mock_llm("nothing useful here")),
        ):
            result = discover_theme()
        assert result == _FALLBACK_THEMES[0]

    def test_existing_signals_appear_in_user_message(self):
        captured_msgs = []

        def fake_invoke(msgs):
            captured_msgs.extend(msgs)
            return MagicMock(content=_VALID_LLM_RESPONSE)

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke

        with (
            patch("pelican.agents.theme_discoverer.search_arxiv_recent", return_value=_SAMPLE_PAPERS),
            patch("pelican.agents.theme_discoverer.list_signals", return_value=["MOM_1_12", "LOW_VOL"]),
            patch("pelican.agents.theme_discoverer._get_llm", return_value=mock_llm),
        ):
            discover_theme()

        user_msg = next(m["content"] for m in captured_msgs if m["role"] == "user")
        assert "MOM_1_12" in user_msg
        assert "LOW_VOL" in user_msg


# ---------------------------------------------------------------------------
# TestThemeDiscovererNode
# ---------------------------------------------------------------------------

class TestThemeDiscovererNode:
    def _make_state(self, **overrides):
        from pelican.agents.graph import coerce_state
        return coerce_state({"theme": "", "run_id": "test-run", **overrides})

    def test_node_writes_theme_to_state(self):
        with patch("pelican.agents.theme_discoverer.discover_theme", return_value="value factor"):
            node = _make_theme_discoverer_node()
            result = node(self._make_state())
        assert result["theme"] == "value factor"

    def test_node_preserves_other_state_fields(self):
        with patch("pelican.agents.theme_discoverer.discover_theme", return_value="momentum"):
            node = _make_theme_discoverer_node()
            result = node(self._make_state(run_id="abc-123"))
        assert result["run_id"] == "abc-123"
        assert result["theme"] == "momentum"

    def test_node_overwrites_existing_theme(self):
        with patch("pelican.agents.theme_discoverer.discover_theme", return_value="reversal"):
            node = _make_theme_discoverer_node()
            result = node(self._make_state(theme="old theme"))
        assert result["theme"] == "reversal"
