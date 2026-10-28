"""Central configuration: study period, horizons, splits, embargo.

These values mirror Section II of the paper. Change them here rather than in
the modules, so a run is fully described by this file plus the raw data.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

# --- prediction setup -------------------------------------------------------

HORIZONS_WEEKS: tuple[int, ...] = (4, 8, 12)
TRAILING_WINDOWS: tuple[int, ...] = (13, 26, 52)
EMBARGO_WEEKS: int = 12

# How many drugs the weekly review list holds (Section III-E).
REVIEW_LIST_SIZE: int = 25


@dataclass(frozen=True)
class Split:
    name: str
    start: dt.date
    end: dt.date


SPLITS: tuple[Split, ...] = (
    Split("train", dt.date(2018, 1, 1), dt.date(2022, 12, 31)),
    Split("val", dt.date(2023, 1, 1), dt.date(2023, 12, 31)),
    Split("test", dt.date(2024, 1, 1), dt.date(2025, 12, 31)),
)

STUDY_START: dt.date = SPLITS[0].start
STUDY_END: dt.date = SPLITS[-1].end


def split_of(week: dt.date) -> str | None:
    """Which split a snapshot week belongs to, before embargo masking."""
    for split in SPLITS:
        if split.start <= week <= split.end:
            return split.name
    return None


def embargo_excludes(week: dt.date) -> bool:
    """True if `week` falls inside a 12-week band around a split boundary.

    A snapshot at week t yields labels at t+4, t+8 and t+12. If any of those
    windows would cross a split boundary, the snapshot is dropped, so no
    training label can overlap validation or test data.
    """
    for split in SPLITS[:-1]:
        boundary = split.end
        if boundary < week <= boundary + dt.timedelta(weeks=EMBARGO_WEEKS):
            return True
        next_start = split.end + dt.timedelta(days=1)
        if next_start - dt.timedelta(weeks=EMBARGO_WEEKS) <= week < next_start:
            return True
    return False


@dataclass(frozen=True)
class Paths:
    root: Path = Path("data")

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def fixtures(self) -> Path:
        return self.root / "fixtures"

    def ensure(self) -> "Paths":
        for path in (self.raw, self.snapshots, self.runs, self.fixtures):
            path.mkdir(parents=True, exist_ok=True)
        return self


# --- data sources -----------------------------------------------------------
# Documented here so every fetched file can be traced to its origin and the
# date it was retrieved.

SOURCES = {
    "ndc": {
        "what": "FDA National Drug Code Directory (ndctext.zip)",
        "url": "https://www.accessdata.fda.gov/cder/ndctext.zip",
    },
    "openfda": {
        "what": "openFDA drug enforcement / recall reports",
        "url": "https://api.fda.gov/drug/enforcement.json",
    },
    "dailymed": {
        "what": "DailyMed Structured Product Labeling by SetID",
        "url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
    },
    "warning_letters": {
        "what": "FDA warning letters (FDA Data Dashboard export or scraped text)",
        "url": "https://datadashboard.fda.gov/ora/index.htm",
    },
    "shortages": {
        "what": "FDA Drug Shortages Database",
        "url": "https://dps.fda.gov/drugshortages",
    },
}
