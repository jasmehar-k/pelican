"""
arXiv search tool for the Researcher agent.

Queries the arXiv REST API (api.arxiv.org/search/) — no API key required.
Searches categories: q-fin.PM (portfolio mgmt), q-fin.ST (statistical finance),
econ.GN, and cs.LG (ML methods applied to finance).

Returns a list of SearchResult dicts: {title, authors, abstract, arxiv_id, url}.
Abstracts are truncated to 800 chars to fit comfortably in the LLM context window.
The Researcher uses these results to ground SignalSpec hypotheses in peer-reviewed
literature before passing to the Coder.

No external API key needed. Rate limit: 1 req/3s per arXiv's terms of service.
"""
