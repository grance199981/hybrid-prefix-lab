"""Shared helpers for the HybridPrefix-Lab experiments.

    client       talk to a vLLM server: streamed requests, /metrics, cache reset
    prompts      synthetic prompts of an exact token length
    sessions     load multi-turn sessions from the three datasets and render them
    cache_model  what the prefix cache should hit, and how many prefill steps
                 the scheduler takes, computed from vLLM's own rules
    stats        small statistics helpers (median, percentile, paired summary, line fit)
    nsys         read an Nsight Systems trace exported to SQLite

Each experiment under experiments/ is a thin layer on top of these modules.
"""
