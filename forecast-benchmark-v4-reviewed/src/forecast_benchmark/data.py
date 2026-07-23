"""Well production data types. No DB, no lateral norm, no crudecode dependency.

A WellSeries is one well's monthly production history. Phases the well
doesn't have data for are left as None-filled arrays rather than faked —
see PHASES and the phase_available flag pattern used throughout this repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHASES = ("oil", "gas", "water")


@dataclass(frozen=True)
class WellSeries:
    """One well's monthly production history, in calendar order, no gaps.

    Arrays are aligned to `months` (same length). A phase with no data at all
    (e.g. water never reported) should be an all-NaN array, not zeros —
    zeros mean "reported zero," NaN means "not reported."
    """
    well_id: str
    months: list[str]          # ISO "YYYY-MM-01" strings, contiguous, ascending
    oil: np.ndarray            # bbl/month
    gas: np.ndarray            # mcf/month
    water: np.ndarray          # bbl/month

    def __post_init__(self) -> None:
        n = len(self.months)
        for name in PHASES:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{self.well_id}: {name} length {len(arr)} != months length {n}")

    def phase_available(self, phase: str) -> bool:
        """True if this phase has at least one non-NaN, non-trivial reading."""
        arr = getattr(self, phase)
        return bool(np.any(~np.isnan(arr)))

    def truncate(self, n_months: int) -> "WellSeries":
        """Point-in-time view: first n_months of history only. Used to build
        the train side of a train/holdout split — never peeks past cutoff."""
        if n_months > len(self.months):
            raise ValueError(f"{self.well_id}: cannot truncate to {n_months}, only {len(self.months)} available")
        return WellSeries(
            well_id=self.well_id,
            months=self.months[:n_months],
            oil=self.oil[:n_months],
            gas=self.gas[:n_months],
            water=self.water[:n_months],
        )


def load_csv(path: str) -> list[WellSeries]:
    """Load wells from a long-format CSV: well_id, month, oil, gas, water.

    Missing water column, or blank cells, become NaN — never 0. One row per
    well-month; wells must be contiguous (no gap months) — this loader does
    not fill gaps, it raises on them, because silently filling gaps hides
    exactly the kind of data-quality issue a benchmark exists to catch.
    """
    import csv as csv_mod
    from collections import defaultdict

    rows_by_well: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows_by_well[row["well_id"]].append(row)

    wells = []
    for well_id, rows in rows_by_well.items():
        rows.sort(key=lambda r: r["month"])
        months = [r["month"] for r in rows]
        _assert_contiguous(well_id, months)
        oil = np.array([_to_float(r.get("oil")) for r in rows])
        gas = np.array([_to_float(r.get("gas")) for r in rows])
        water = np.array([_to_float(r.get("water")) for r in rows])
        wells.append(WellSeries(well_id=well_id, months=months, oil=oil, gas=gas, water=water))
    return wells


def _to_float(v: str | None) -> float:
    if v is None or v == "":
        return float("nan")
    return float(v)


def _assert_contiguous(well_id: str, months: list[str]) -> None:
    from datetime import date
    for a, b in zip(months, months[1:]):
        ya, ma = int(a[:4]), int(a[5:7])
        yb, mb = int(b[:4]), int(b[5:7])
        expected = (ya + (ma // 12), (ma % 12) + 1) if ma == 12 else (ya, ma + 1)
        if (yb, mb) != expected:
            raise ValueError(f"{well_id}: gap or disorder between {a} and {b} — this loader does not fill gaps")
