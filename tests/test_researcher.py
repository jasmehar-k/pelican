from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from pelican.agents.graph import initial_state
from pelican.agents.researcher import _make_researcher_node, get_hypotheses
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
        assert "abs:momentum" in mock_get.call_args.kwargs["params"]["search_query"]

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
        assert "cat:q-fin.TR" in query

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


class TestGetHypotheses:
    """Tests for the multi-signal get_hypotheses() public function."""

    PAPERS = [
        {
            "title": "Momentum and Drift",
            "authors": ["Alice Smith"],
            "abstract": "Signals built from post-earnings drift can persist.",
            "arxiv_id": "2401.00001",
            "url": "https://arxiv.org/abs/2401.00001",
        }
    ]

    def _mock_llm(self, text: str):
        msg = MagicMock()
        msg.content = text
        llm = MagicMock()
        llm.invoke.return_value = msg
        return llm

    def test_returns_n_hypotheses(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        response = (
            "HYPOTHESIS_1: Earnings revision breadth predicts future returns because analysts herd toward consensus, causing prices to drift in the direction of revisions.\n"
            "DATA_FIELDS_1: close, close_21d\n"
            "SIGNAL_NAME_1: earnings_revision\n\n"
            "HYPOTHESIS_2: Return on equity captures how efficiently a firm converts equity into profit, with high-ROE firms rewarded by the market over time.\n"
            "DATA_FIELDS_2: roe\n"
            "SIGNAL_NAME_2: quality_roe\n\n"
            "HYPOTHESIS_3: Low volatility stocks earn higher risk-adjusted returns because institutional investors overpay for lottery-like high-vol names.\n"
            "DATA_FIELDS_3: vol_21d\n"
            "SIGNAL_NAME_3: low_vol\n"
        )
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=self.PAPERS),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=self._mock_llm(response)),
        ):
            papers, hypotheses = get_hypotheses("earnings quality", n=3)

        assert papers == self.PAPERS
        assert len(hypotheses) == 3
        assert hypotheses[0]["signal_name"] == "earnings_revision"
        assert hypotheses[1]["signal_name"] == "quality_roe"
        assert hypotheses[2]["signal_name"] == "low_vol"
        assert "roe" in hypotheses[1]["data_fields"]

    def test_each_hypothesis_has_required_keys(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        response = (
            "HYPOTHESIS_1: Stocks with strong recent returns tend to continue outperforming due to momentum.\n"
            "DATA_FIELDS_1: close_252d, close_21d\n"
            "SIGNAL_NAME_1: momentum\n"
        )
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=self.PAPERS),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=self._mock_llm(response)),
        ):
            _, hypotheses = get_hypotheses("momentum", n=1)

        h = hypotheses[0]
        assert "hypothesis" in h
        assert "data_fields" in h
        assert "signal_name" in h
        assert isinstance(h["data_fields"], list)

    def test_empty_papers_returns_no_hypotheses(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        with patch("pelican.agents.researcher.search_arxiv", return_value=[]):
            papers, hypotheses = get_hypotheses("anything", n=3)

        assert papers == []
        assert hypotheses == []

    def test_papers_stored_in_vector_store(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        response = "HYPOTHESIS_1: x\nDATA_FIELDS_1: close\nSIGNAL_NAME_1: s\n"
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=self.PAPERS),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=self._mock_llm(response)),
            patch("pelican.agents.researcher.store_paper") as mock_store,
            patch("pelican.agents.researcher.has_paper", return_value=False),
        ):
            get_hypotheses("theme", n=1)

        assert mock_store.called

    def test_multi_user_message_requests_n_hypotheses(self, tmp_path, monkeypatch):
        _configure_env(monkeypatch, tmp_path)
        captured = []

        def fake_invoke(messages):
            captured.extend(messages)
            msg = MagicMock()
            msg.content = "HYPOTHESIS_1: x\nDATA_FIELDS_1: close\nSIGNAL_NAME_1: s\n"
            return msg

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = fake_invoke
        with (
            patch("pelican.agents.researcher.search_arxiv", return_value=self.PAPERS),
            patch("pelican.agents.researcher.find_similar", return_value=[]),
            patch("pelican.agents.researcher._get_llm", return_value=mock_llm),
        ):
            get_hypotheses("earnings quality", n=3)

        user_msg = captured[1]["content"]
        assert "HYPOTHESIS_1" in user_msg
        assert "HYPOTHESIS_2" in user_msg
        assert "HYPOTHESIS_3" in user_msg
        assert "3" in user_msg


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
