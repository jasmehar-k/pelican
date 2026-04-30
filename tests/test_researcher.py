from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from pelican.agents.graph import initial_state
from pelican.agents.researcher import _make_researcher_node
from pelican.agents.tools.pdf_extract import fetch_pdf_text
from pelican.agents.tools.search import search_arxiv
from pelican.agents.tools.vector_store import find_similar, has_paper, store_paper
from pelican.data.store import DataStore
from pelican.utils.config import get_settings


ATOM_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Momentum and Drift</title>
    <summary>Signals built from post-earnings drift can persist when ranking is delayed.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Quality and Leverage</title>
    <summary>Balance-sheet strength matters for cross-sectional equity returns.</summary>
    <author><name>Carol Lee</name></author>
  </entry>
</feed>
"""

LONG_ABSTRACT = "x" * 900


@pytest.fixture()
def store() -> DataStore:
    s = DataStore(":memory:")
    s.init_schema()
    return s


def _configure_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "db.duckdb"))
    get_settings.cache_clear()


class TestArxivSearch:
    def test_parse_atom_returns_results(self):
        response = MagicMock()
        response.text = ATOM_FEED
        response.raise_for_status.return_value = None

        with patch("pelican.agents.tools.search.httpx.get", return_value=response) as mock_get:
            result = search_arxiv("momentum", max_results=2)

        assert len(result) == 2
        assert result[0]["title"] == "Momentum and Drift"
        assert result[0]["authors"] == ["Alice Smith", "Bob Jones"]
        assert result[0]["arxiv_id"] == "2401.00001"
        assert result[0]["url"].endswith("2401.00001")
        assert mock_get.call_args.kwargs["params"]["search_query"].startswith("(momentum)")

    def test_abstract_truncated_to_800_chars(self):
        feed = ATOM_FEED.replace(
            "Signals built from post-earnings drift can persist when ranking is delayed.",
            LONG_ABSTRACT,
        )
        response = MagicMock()
        response.text = feed
        response.raise_for_status.return_value = None

        with patch("pelican.agents.tools.search.httpx.get", return_value=response):
            result = search_arxiv("momentum")

        assert len(result[0]["abstract"]) == 800

    def test_empty_feed_returns_empty_list(self):
        response = MagicMock()
        response.text = "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
        response.raise_for_status.return_value = None

        with patch("pelican.agents.tools.search.httpx.get", return_value=response):
            result = search_arxiv("value")

        assert result == []

    def test_rate_limit_sleep_called(self):
        response = MagicMock()
        response.text = ATOM_FEED
        response.raise_for_status.return_value = None

        with (
            patch("pelican.agents.tools.search.httpx.get", return_value=response),
            patch("pelican.agents.tools.search.time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.1]),
            patch("pelican.agents.tools.search.time.sleep") as mock_sleep,
        ):
            search_arxiv("momentum")
            search_arxiv("momentum")

        assert mock_sleep.called

    def test_query_includes_category_filter(self):
        response = MagicMock()
        response.text = ATOM_FEED
        response.raise_for_status.return_value = None

        with patch("pelican.agents.tools.search.httpx.get", return_value=response) as mock_get:
            search_arxiv("earnings quality")

        query = mock_get.call_args.kwargs["params"]["search_query"]
        assert "cat:q-fin.PM" in query
        assert "cat:cs.LG" in query

    def test_http_error_raises(self):
        with patch("pelican.agents.tools.search.httpx.get", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                search_arxiv("momentum")


class TestPdfExtract:
    def test_extracts_text_from_first_five_pages(self):
        page = MagicMock()
        page.extract_text.side_effect = ["page 1", "page 2", "page 3", "page 4", "page 5"]
        reader = MagicMock()
        reader.pages = [page, page, page, page, page, page]

        class FakePdfReader:
            def __init__(self, *_args, **_kwargs):
                self.pages = reader.pages

        response = MagicMock()
        response.content = b"pdf"
        response.raise_for_status.return_value = None

        with (
            patch("pelican.agents.tools.pdf_extract.httpx.get", return_value=response),
            patch("pelican.agents.tools.pdf_extract.PdfReader", FakePdfReader),
        ):
            text = fetch_pdf_text("2401.00001")

        assert text == "page 1\npage 2\npage 3\npage 4\npage 5"

    def test_returns_empty_string_on_http_failure(self):
        with patch("pelican.agents.tools.pdf_extract.httpx.get", side_effect=RuntimeError("boom")):
            assert fetch_pdf_text("2401.00001") == ""

    def test_returns_empty_string_on_parse_failure(self):
        response = MagicMock()
        response.content = b"broken"
        response.raise_for_status.return_value = None

        with patch("pelican.agents.tools.pdf_extract.httpx.get", return_value=response):
            assert fetch_pdf_text("2401.00001") == ""


class TestVectorStore:
    def test_store_and_has_paper(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        store_paper("2401.00001", "Title", "Abstract", {"url": "https://arxiv.org/abs/2401.00001"})
        assert has_paper("2401.00001") is True

    def test_unknown_paper_not_in_store(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        assert has_paper("9999.99999") is False

    def test_find_similar_returns_near_duplicate(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        store_paper("2401.00001", "Title", "Abstract about momentum signals", {"url": "u"})
        matches = find_similar("Abstract about momentum signals", threshold=0.0)
        assert matches
        assert matches[0]["similarity"] >= 0.0

    def test_find_similar_empty_collection_returns_empty(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        assert find_similar("anything") == []


class TestResearcherNode:
    def _mock_llm_response(self, text: str):
        msg = MagicMock()
        msg.content = text
        return msg

    def _state(self, **overrides):
        state = initial_state("earnings momentum")
        state.update(overrides)
        return state

    def test_researcher_enriches_theme(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        papers = [
            {
                "title": "Momentum and Drift",
                "authors": ["Alice Smith"],
                "abstract": "Signals built from post-earnings drift can persist.",
                "arxiv_id": "2401.00001",
                "url": "https://arxiv.org/abs/2401.00001",
            },
            {
                "title": "Quality and Leverage",
                "authors": ["Carol Lee"],
                "abstract": "Balance-sheet strength matters.",
                "arxiv_id": "2401.00002",
                "url": "https://arxiv.org/abs/2401.00002",
            },
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "HYPOTHESIS: Use delayed momentum after earnings announcements.\n"
            "DATA_FIELDS: close, close_21d, close_252d\n"
            "SIGNAL_NAME: post_earnings_momentum\n"
            "NEED_MORE_DETAIL: false\n"
        )

        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=papers),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
        ):
            node = _make_researcher_node()
            result = node(self._state())

        assert result["signal_hypothesis"]
        assert result["papers"] == papers
        assert result["arxiv_ids"] == ["2401.00001", "2401.00002"]
        assert mock_llm.invoke.call_count == 1

    def test_papers_list_populated(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        paper = {
            "title": "Momentum and Drift",
            "authors": ["Alice Smith"],
            "abstract": "Signals built from post-earnings drift can persist.",
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "HYPOTHESIS: Use delayed momentum.\nDATA_FIELDS: close_21d\nSIGNAL_NAME: x\nNEED_MORE_DETAIL: false"
        )

        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=[paper]),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
        ):
            result = _make_researcher_node()(self._state())

        assert result["papers"]

    def test_arxiv_ids_stored_in_state(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        paper = {
            "title": "Momentum and Drift",
            "authors": ["Alice Smith"],
            "abstract": "Signals built from post-earnings drift can persist.",
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "HYPOTHESIS: Use delayed momentum.\nDATA_FIELDS: close_21d\nSIGNAL_NAME: x\nNEED_MORE_DETAIL: false"
        )
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=[paper]),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
        ):
            result = _make_researcher_node()(self._state())

        assert result["arxiv_ids"] == ["2401.00001"]

    def test_researcher_falls_back_to_theme_if_no_papers(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        with patch("pelican.agents.researcher.search_arxiv", return_value=[]):
            result = _make_researcher_node()(self._state())

        assert result["signal_hypothesis"] is None
        assert result["theme"] == "earnings momentum"
        assert result["papers"] == []

    def test_papers_stored_in_vector_store(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        paper = {
            "title": "Momentum and Drift",
            "authors": ["Alice Smith"],
            "abstract": "Signals built from post-earnings drift can persist.",
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "HYPOTHESIS: Use delayed momentum.\nDATA_FIELDS: close_21d\nSIGNAL_NAME: x\nNEED_MORE_DETAIL: false"
        )
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=[paper]),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
            patch("pelican.agents.researcher.store_paper") as mock_store,
        ):
            _make_researcher_node()(self._state())

        assert mock_store.called

    def test_duplicate_papers_filtered(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        paper = {
            "title": "Momentum and Drift",
            "authors": ["Alice Smith"],
            "abstract": "Signals built from post-earnings drift can persist.",
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_llm_response(
            "HYPOTHESIS: Use delayed momentum.\nDATA_FIELDS: close_21d\nSIGNAL_NAME: x\nNEED_MORE_DETAIL: false"
        )
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=[paper]),
            patch("pelican.agents.researcher.find_similar", return_value=[{"arxiv_id": "old", "similarity": 0.99}]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
        ):
            result = _make_researcher_node()(self._state())

        assert result["papers"] == [paper]
        assert result["arxiv_ids"] == ["2401.00001"]

    def test_run_id_set_in_initial_state(self):
        state = initial_state("theme")
        uuid.UUID(state["run_id"])


class TestResearchLog:
    def test_log_run_inserts_row(self):
        store = DataStore(":memory:")
        store.init_schema()
        state = {
            "run_id": "run-1",
            "theme": "earnings momentum",
            "arxiv_ids": ["2401.00001"],
            "signal_hypothesis": "Momentum after earnings",
            "generated_code": "code",
            "decision": "accept",
            "ic_tstat": 2.5,
            "sharpe_net": 0.7,
            "feedback": "ok",
        }
        store.log_run(state)
        result = store.query("SELECT * FROM research_log")
        assert result.shape[0] == 1
        assert result["run_id"][0] == "run-1"
        assert result["theme"][0] == "earnings momentum"
        assert result["arxiv_ids"].to_list()[0] == ["2401.00001"]
        assert result["decision"][0] == "accept"

    def test_research_log_table_created(self):
        store = DataStore(":memory:")
        store.init_schema()
        tables = store.query("SHOW TABLES")
        assert "research_log" in tables["name"].to_list()
