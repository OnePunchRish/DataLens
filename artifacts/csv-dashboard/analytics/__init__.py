"""
analytics — DataLens analysis package.

Exports the three public entry points used by the Flask routes:
  - build_charts       → chart JSON for Plotly
  - generate_insights  → column-level stats and auto-notes
  - generate_ai_insights → rule-based AI insight cards
  - answer_question    → natural-language query engine
"""

from .charts   import build_charts
from .insights import generate_insights, generate_ai_insights
from .query    import answer_question

__all__ = [
    "build_charts",
    "generate_insights",
    "generate_ai_insights",
    "answer_question",
]
