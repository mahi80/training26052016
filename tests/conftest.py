"""Shared pytest fixtures: make imports work and guarantee data artifacts exist.

WHY THIS EXISTS
---------------
Two problems every test in this project would otherwise hit:

1. **Imports**: tests live in ``tests/`` but import ``src...`` and ``data...``
   packages from the project root. Rather than require an editable install, we
   prepend the project root to ``sys.path`` here — conftest.py is imported by
   pytest before any test module, so every test sees the right path.

2. **Data**: most tests need the generated artifacts (CSV, contracts, warehouse,
   catalog). The session-scoped autouse fixture below runs the relevant generator's
   ``main()`` for any artifact that is missing, exactly once per test session. The
   generators are deterministic (seed 42) and import-safe (no side effects at import
   time), so this is both fast and reproducible. Artifacts already on disk are
   reused — per ARCHITECTURE.md §6 we generate into the real ``data/`` dir once,
   not into a tmp dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402  (sys.path must be patched before project imports)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers (no pytest.ini — the directory layout in
    ARCHITECTURE.md §2 is fixed, so configuration lives here instead)."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (e.g. end-to-end model training); "
        "deselect with `-m 'not slow'`",
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_data() -> None:
    """Generate any missing data artifact once per test session (offline, seeded)."""
    from data import build_database, generate_contracts, generate_supply_chain_data

    csv_path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"
    contracts_dir = PROJECT_ROOT / "data" / "contracts"
    db_path = PROJECT_ROOT / "data" / "warehouse.db"
    catalog_path = PROJECT_ROOT / "data" / "metadata_catalog.json"

    if not csv_path.exists():
        generate_supply_chain_data.main()
    if not contracts_dir.exists() or not any(contracts_dir.glob("*.md")):
        generate_contracts.main()
    if not db_path.exists() or not catalog_path.exists():
        build_database.main()
