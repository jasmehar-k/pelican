"""
News sentiment data ingestion.

Fetches and preprocesses news articles and earnings call transcripts.
Produces per-ticker daily sentiment scores (positive/negative/neutral)
using an LLM or lightweight classifier. Scores are stored with the
publication timestamp as the point-in-time anchor.
"""
