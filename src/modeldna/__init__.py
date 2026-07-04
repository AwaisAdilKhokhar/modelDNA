"""modeldna — forensic lineage detection for open-weight LLMs.

Fingerprints a model from its weights (via cheap partial reads) and compares
it against a reference database of known base models to produce a calibrated
lineage verdict.
"""

__version__ = "0.1.0"
