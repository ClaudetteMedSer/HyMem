"""Backward-compatible shim.

The Honcho-compatible server was split into the ``hymem.honcho`` package
(models / adapters / app). This module re-exports the public surface so any
existing ``hymem.honcho_server`` import keeps working.
"""
from hymem.honcho.app import app, main, set_hy, set_scheduler

__all__ = ["app", "main", "set_hy", "set_scheduler"]
