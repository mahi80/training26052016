"""OpenMetadata-style catalog client: schema + glossary grounding for NL2SQL.

WHY THIS EXISTS
---------------
The single biggest failure mode of NL2SQL is an LLM hallucinating tables and
columns. The fix is not a better prompt trick — it is *grounding*: before drafting
SQL, the agent reads an authoritative description of the warehouse from a
**metadata catalog**. This module is a miniature, file-backed version of what
tools like OpenMetadata, DataHub, or Unity Catalog provide in production:

- tables with business descriptions ("fact table, grain = order_id"),
- per-column descriptions and tags (PK / FK / target / ml-leakage),
- a business glossary mapping vague terms ("late delivery") to concrete columns,
- vetted sample queries the agent can imitate.

``MetadataCatalog`` reads ``data/metadata_catalog.json`` (built by
``data/build_database.py``, shaped like OpenMetadata's Table/Glossary entities)
and renders it as *prompt-ready text*. The NL2SQL tools in
``src/agents/tools_sql.py`` are thin wrappers over this class.

WHY A CATALOG IS THE TOY-VS-PRODUCTION LINE FOR NL2SQL
------------------------------------------------------
1. **Schema drift**: production schemas change weekly; a catalog is refreshed by
   ingestion pipelines, so the agent always reasons over the *current* schema —
   a schema string pasted into a prompt rots silently.
2. **Column semantics + governance tags**: ``on_time_sla_pct`` means nothing to an
   LLM without its description; PII/leakage tags let you block or mask columns the
   agent must never SELECT (here: the ``ml-leakage`` tags on outcome columns).
3. **Query examples**: curated ``sampleQueries`` are few-shot gold — they teach the
   join paths and idioms of *your* warehouse far better than generic SQL training.

PRODUCTION PATH — same interface, backed by a real OpenMetadata server
----------------------------------------------------------------------
In an engagement you would not ship a JSON file; you would point this class at the
client's OpenMetadata instance via the ``openmetadata-ingestion`` SDK. The public
interface below stays identical — only ``__init__`` and the data access change::

    # pip install openmetadata-ingestion
    # import os
    # from metadata.ingestion.ometa.ometa_api import OpenMetadata
    # from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
    #     OpenMetadataConnection, AuthProvider,
    # )
    # from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
    #     OpenMetadataJWTClientConfig,
    # )
    # from metadata.generated.schema.entity.data.table import Table
    #
    # class MetadataCatalog:                       # same public interface
    #     def __init__(self) -> None:
    #         self._om = OpenMetadata(OpenMetadataConnection(
    #             hostPort=os.environ["OPENMETADATA_HOST_PORT"],   # e.g. https://meta.acme.com/api
    #             authProvider=AuthProvider.openmetadata,
    #             securityConfig=OpenMetadataJWTClientConfig(
    #                 jwtToken=os.environ["OPENMETADATA_JWT_TOKEN"],
    #             ),
    #         ))
    #
    #     def list_tables(self) -> list[str]:
    #         listed = self._om.list_entities(entity=Table, fields=["name"], limit=100)
    #         return [t.name.root for t in listed.entities]
    #
    #     def get_table_schema(self, table: str) -> str:
    #         t = self._om.get_by_name(                            # fully-qualified name lookup
    #             entity=Table,
    #             fqn=f"supply_chain_warehouse.main.{table}",
    #             fields=["columns", "tags", "description"],
    #         )
    #         ...render t.columns / t.description exactly as below...
    #
    #     def search(self, query: str) -> list[str]:
    #         hits = self._om.es_search_from_fqn(entity_type=Table, fqn_search_string=query)
    #         ...rank and return table names...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

DEFAULT_CATALOG_PATH: Path = PROJECT_ROOT / "data" / "metadata_catalog.json"

# Relevance weights for keyword search — name hits matter most, then column names,
# then prose mentions. Tuned for explainability, not IR perfection.
_W_TABLE_NAME = 5.0
_W_COLUMN_NAME = 3.0
_W_TAG = 2.0
_W_TABLE_DESC = 2.0
_W_COLUMN_DESC = 1.0


class MetadataCatalog:
    """Read-only client over the OpenMetadata-style catalog JSON.

    Every method returns plain text (or names) ready to drop into an NL2SQL
    prompt — callers never need to parse the catalog JSON themselves.
    """

    def __init__(self, catalog_path: Path | None = None) -> None:
        self.catalog_path: Path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Metadata catalog not found at {self.catalog_path}. "
                "Run `python data/generate_supply_chain_data.py` then "
                "`python data/build_database.py` from the project root."
            )
        self._catalog: dict[str, Any] = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self._tables: dict[str, dict[str, Any]] = {
            t["name"]: t for t in self._catalog.get("tables", [])
        }

    # ------------------------------------------------------------------ basics

    def list_tables(self) -> list[str]:
        """All table names, in catalog order."""
        return list(self._tables.keys())

    def get_table_description(self, table: str) -> str:
        """The business description of one table (raises ``KeyError`` if unknown)."""
        if table not in self._tables:
            raise KeyError(
                f"Unknown table {table!r}. Available tables: {', '.join(self.list_tables())}"
            )
        return str(self._tables[table].get("description", ""))

    @property
    def database_path(self) -> Path:
        """Resolve the warehouse file from the catalog's service entry.

        (Extension beyond the §4.5 contract: lets ``tools_sql`` discover the DB
        through the catalog instead of hardcoding it — exactly how a production
        agent would resolve a connection from OpenMetadata's service metadata.)
        """
        relative = self._catalog.get("service", {}).get("database", "data/warehouse.db")
        return PROJECT_ROOT / relative

    # ------------------------------------------------------------------ schema

    def get_table_schema(self, table: str) -> str:
        """DDL-ish text for one table: columns, types, tags, descriptions, samples.

        Raises ``KeyError`` (listing valid names) for unknown tables — the @tool
        wrapper converts that into a self-repair hint for the agent.
        """
        if table not in self._tables:
            raise KeyError(
                f"Unknown table {table!r}. Available tables: {', '.join(self.list_tables())}"
            )
        entry = self._tables[table]
        lines: list[str] = []
        tags = ", ".join(entry.get("tags", []))
        lines.append(f"-- Table: {entry['name']}" + (f"  [tags: {tags}]" if tags else ""))
        lines.append(f"-- {entry['description']}")
        lines.append(f"CREATE TABLE {entry['name']} (")
        columns = entry.get("columns", [])
        for i, col in enumerate(columns):
            comma = "," if i < len(columns) - 1 else ""
            col_tags = "".join(f"[{t}] " for t in col.get("tags", []))
            lines.append(
                f"    {col['name']} {col['dataType']}{comma}  -- {col_tags}{col['description']}"
            )
        lines.append(");")
        samples = entry.get("sampleQueries", [])
        if samples:
            lines.append("-- Sample queries (vetted — imitate their join paths):")
            lines.extend(f"--   {q}" for q in samples)
        return "\n".join(lines)

    def get_full_schema(self) -> str:
        """Every table's schema plus the glossary — paste-ready NL2SQL grounding."""
        service = self._catalog.get("service", {})
        header = (
            f"-- Warehouse: {service.get('name', 'unknown')} "
            f"({service.get('type', '?')}, file: {service.get('database', '?')})\n"
            f"-- Tables: {', '.join(self.list_tables())}"
        )
        schemas = [self.get_table_schema(name) for name in self.list_tables()]
        return "\n\n".join([header, *schemas, self.get_glossary()])

    # ------------------------------------------------------------------ search

    def search(self, query: str) -> list[str]:
        """Table names ranked by keyword relevance to ``query``.

        Scoring is intentionally transparent (weighted substring counts over table
        name > column names > tags > descriptions) so trainees can predict and
        debug the ranking. Tables with zero relevance are dropped; if *nothing*
        matches, all tables are returned in catalog order so the agent always has
        somewhere to look next.
        """
        tokens = [t for t in re.findall(r"[a-z0-9_]+", query.lower()) if len(t) >= 2]
        scores: dict[str, float] = {}
        for name, entry in self._tables.items():
            score = 0.0
            table_name = name.lower()
            table_desc = str(entry.get("description", "")).lower()
            for tok in tokens:
                if tok in table_name:
                    score += _W_TABLE_NAME
                score += _W_TABLE_DESC * table_desc.count(tok)
                score += _W_TAG * sum(1 for tag in entry.get("tags", []) if tok in tag.lower())
                for col in entry.get("columns", []):
                    if tok in col["name"].lower():
                        score += _W_COLUMN_NAME
                    score += _W_COLUMN_DESC * str(col.get("description", "")).lower().count(tok)
                    score += _W_TAG * sum(
                        1 for tag in col.get("tags", []) if tok in tag.lower()
                    )
            scores[name] = score
        ranked = sorted(self.list_tables(), key=lambda n: -scores[n])  # stable sort
        matched = [n for n in ranked if scores[n] > 0]
        return matched if matched else self.list_tables()

    # ---------------------------------------------------------------- glossary

    def get_glossary(self) -> str:
        """The business glossary as prompt-ready text (term → definition → columns)."""
        lines = ["-- Business glossary (terms users say → columns that mean them):"]
        for entry in self._catalog.get("glossary", []):
            assets = ", ".join(entry.get("mappedAssets", []))
            lines.append(f"--   {entry['term']}: {entry['definition']} [maps to: {assets}]")
        return "\n".join(lines)
