"""Backward-compatible shim.

The Honcho v3-compatible server was split into the ``hymem.honcho`` package
(models / adapters / app). This module re-exports the public surface so any
existing ``hymem.honcho_server`` import keeps working.
"""
from hymem.honcho.app import app, main

__all__ = ["app", "main"]
