"""OpenMetadata-style catalog package for the supply-chain warehouse.

WHY THIS EXISTS
---------------
NL2SQL agents fail when they guess at schemas. This package wraps the project's
metadata catalog (``data/metadata_catalog.json``) behind one small client class so
every consumer — the SQL tools, the labs, the prompts — asks the *catalog* for
schema text instead of hardcoding DDL strings. Swap the JSON for a real
OpenMetadata server (see ``catalog.py``) and nothing downstream changes.
"""

from __future__ import annotations

from .catalog import MetadataCatalog

__all__ = ["MetadataCatalog"]
