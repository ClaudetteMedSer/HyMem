"""Honcho v3-compatible HTTP server for HyMem.

Package layout:
  models.py    — typed Pydantic request models (one per endpoint body)
  adapters.py  — response shaping + request-shape normalization
  app.py       — FastAPI routes, background dreaming, entry point

Import the FastAPI instance and entry point from ``hymem.honcho.app`` directly
(this ``__init__`` deliberately stays import-free so the ``app`` submodule is
not shadowed by an ``app`` attribute).
"""
