"""Inference for pruned networks: compaction, a CSR sparse path, and cost measurement.

Deliberately imports nothing at package level, so `python -m infer.benchmark`
can pin BLAS threads before NumPy is first imported.
"""
