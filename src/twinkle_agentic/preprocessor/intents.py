"""Intent category constants (AUDIT A4).

Sunk out of ``intent_classifier.py`` into this dependency-free module so any
consumer (e.g. the experimental log-prob scorers, or downstream sampling that
reads ``intents`` labels) can reference the vocabulary without importing the
heavier classifier + its regex detectors.
"""

INTENT_TOOL_CALL = 'tool_call'
INTENT_CODE = 'code'
INTENT_MATH = 'math'
INTENT_COMPLEX_LOGIC = 'complex_logic'
INTENT_REASONING = 'reasoning'
INTENT_USER_DISSATISFACTION = 'user_dissatisfaction'
INTENT_OTHER = 'other'

ALL_INTENTS = (
    INTENT_TOOL_CALL,
    INTENT_CODE,
    INTENT_MATH,
    INTENT_COMPLEX_LOGIC,
    INTENT_REASONING,
    INTENT_USER_DISSATISFACTION,
    INTENT_OTHER,
)
