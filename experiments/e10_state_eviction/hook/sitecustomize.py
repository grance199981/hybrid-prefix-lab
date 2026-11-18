"""Loaded by Python's `site` at start-up when this directory is on PYTHONPATH (run_e10.sh
puts it there for the vLLM server only). Does nothing unless E10_HOOK_DIR is set."""
import os

if os.environ.get("E10_HOOK_DIR"):
    import e10_hook

    e10_hook.install()
