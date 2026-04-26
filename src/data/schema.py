"""Data schema definitions for LLM Safety Classifier."""

import pandas as pd

# Expected columns in corpus.parquet
CORPUS_COLUMNS = ["text", "label", "source", "meta"]

# Valid sources
VALID_SOURCES = [
    "xguard",
    "beavertails",
    "toxicchat",
    "xstest",
    "hh-rlhf",
    "do-not-answer",
    "aegis",
    "wildguard",
]

# Label mapping
LABEL_SAFE = 0
LABEL_UNSAFE = 1
