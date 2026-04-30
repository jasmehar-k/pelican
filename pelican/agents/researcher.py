"""
Researcher agent node.

Given a research theme (e.g. "earnings quality factors"), searches academic
databases and the web for alpha signal ideas. Outputs a structured SignalSpec:
name, hypothesis, required data fields, expected holding period, and relevant
citations. Runs as a LangGraph node; receives and returns the pipeline State.
"""
