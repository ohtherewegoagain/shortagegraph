"""Graph construction, features and labels (Sections II-A to II-C).

A snapshot is built from records publicly available on or before its week, so
no feature can leak from the future. The unit of prediction is a *drug node*:
all NDC products sharing an active ingredient set, dosage form and route.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import HORIZONS_WEEKS, TRAILING_WINDOWS, embargo_excludes, split_of
from .ingest import NdcProduct, RecallReport, ShortagePosting, WarningLetter, normalise_firm

DRUG = 0
LABELER = 1
ESTAB = 2

# Feature names are kept as lists so downstream code can index by name rather
# than by position, and so a results table can say what a column means.
FEATURE_NAMES_DRUG = [
    "n_labelers",
    "n_establishments",
    "hhi_labelers",
    "share_estab_outside_us",
    "sterile_injectable",
    "recall_I_13w",
    "recall_II_13w",
    "recall_III_13w",
    "recall_I_26w",
    "recall_II_26w",
    "recall_III_26w",
    "recall_I_52w",
    "recall_II_52w",
    "recall_III_52w",
    "weeks_since_recall",
    "letters_13w",
    "letters_26w",
    "letters_52w",
    "weeks_since_letter",
    "cgmp_categories",
    "import_alert",
]

FEATURE_NAMES_LABELER = [
    "recall_I_26w",
    "recall_II_26w",
    "recall_III_26w",
    "letters_26w",
    "weeks_since_recall",
    "weeks_since_letter",
    "n_products",
]

FEATURE_NAMES_ESTAB = [
    "recall_26w",
    "letters_26w",
    "weeks_since_letter",
    "is_us",
]

RELATIONS = (
    "markets",
    "markets_rev",
    "made_at",
    "made_at_rev",
    "manufactured_for",
    "manufactured_for_rev",
    "substitutes",
)


def drug_key(ingredients, form: str, route: str) -> str:
    """Stable key: sorted ingredients joined by '+', then form, then route."""
    if isinstance(ingredients, (set, frozenset, list, tuple)):
        joined = "+".join(sorted(x.strip().upper() for x in ingredients if x.strip()))
    else:
        joined = "+".join(sorted(x.strip().upper() for x in str(ingredients).split("+") if x.strip()))
    return f"{joined}|{form.strip().upper()}|{route.strip().upper()}"


@dataclass
class RawData:
    """Everything the graph builder needs, independent of where it came from."""

    products: list[NdcProduct]
    establishments: dict[str, dict] = field(default_factory=dict)  # fei -> {name, country}
    product_estabs: dict[str, set[str]] = field(default_factory=dict)  # ndc -> FEIs
    recalls: list[RecallReport] = field(default_factory=list)
    letters: list[WarningLetter] = field(default_factory=list)
    shortages: list[ShortagePosting] = field(default_factory=list)


@dataclass
class Snapshot:
    week: dt.date
    drug_keys: list[str]
    labeler_ids: list[str]
    estab_ids: list[str]
    Xd: np.ndarray
    Xl: np.ndarray
    Xe: np.ndarray
    edges: dict[str, np.ndarray]
    split: str | None = None
    embargoed: bool = False

    @property
    def num_nodes(self) -> int:
        return len(self.drug_keys) + len(self.labeler_ids) + len(self.estab_ids)

    @property
    def num_drugs(self) -> int:
        return len(self.drug_keys)


def iter_weeks(start: dt.date, end: dt.date) -> list[dt.date]:
    """Mondays from `start` to `end`, inclusive."""
    monday = start - dt.timedelta(days=start.weekday())
    weeks: list[dt.date] = []
    while monday <= end:
        weeks.append(monday)
        monday += dt.timedelta(weeks=1)
    return weeks


def _active(product: NdcProduct, week: dt.date) -> bool:
    if product.marketing_start and product.marketing_start > week:
        return False
    if product.marketing_end and product.marketing_end < week:
        return False
    return True


def _weeks_between(earlier: dt.date, later: dt.date) -> float:
    return (later - earlier).days / 7.0


def _ndc9(code: str) -> str:
    digits = "".join(ch for ch in code if ch.isdigit())
    return digits[:9]


def build_snapshot(raw: RawData, week: dt.date) -> Snapshot:
    """Assemble one weekly snapshot from records dated on or before `week`."""
    active = [p for p in raw.products if _active(p, week)]

    drugs: dict[str, list[NdcProduct]] = defaultdict(list)
    for product in active:
        drugs[drug_key(product.ingredient_set, product.dosage_form, product.route)].append(product)
    drug_keys = sorted(drugs)
    drug_index = {key: i for i, key in enumerate(drug_keys)}

    labelers = sorted({p.labeler_code for p in active})
    labeler_index = {name: i for i, name in enumerate(labelers)}

    estabs: set[str] = set()
    for product in active:
        estabs.update(raw.product_estabs.get(product.ndc, set()))
    estab_ids = sorted(estabs)
    estab_index = {fei: i for i, fei in enumerate(estab_ids)}

    n_d, n_l, n_e = len(drug_keys), len(labelers), len(estab_ids)
    offset_l = n_d
    offset_e = n_d + n_l

    # --- recalls linked to products, labelers and drugs ---------------------
    ndc_to_products: dict[str, list[NdcProduct]] = defaultdict(list)
    for product in active:
        ndc_to_products[_ndc9(product.ndc)].append(product)
    labeler_by_name = defaultdict(list)
    for name in labelers:
        labeler_by_name[normalise_firm(name)].append(name)

    # per-drug / per-labeler recall dates by class
    drug_recalls: dict[int, list[tuple[dt.date, str]]] = defaultdict(list)
    labeler_recalls: dict[int, list[tuple[dt.date, str]]] = defaultdict(list)

    for recall in raw.recalls:
        if recall.report_date > week:
            continue
        targets = []
        for code in recall.ndc_codes:
            targets.extend(ndc_to_products.get(_ndc9(code), []))
        if not targets:
            targets = [p for name in labeler_by_name.get(normalise_firm(recall.firm), [])
                       for p in active if p.labeler_code == name]
        for product in targets:
            di = drug_index.get(drug_key(product.ingredient_set, product.dosage_form, product.route))
            li = labeler_index.get(product.labeler_code)
            if di is not None:
                drug_recalls[di].append((recall.report_date, recall.classification))
            if li is not None:
                labeler_recalls[li].append((recall.report_date, recall.classification))

    # --- warning letters linked to establishments --------------------------
    letters_by_estab: dict[int, list[WarningLetter]] = defaultdict(list)
    for letter in raw.letters:
        if letter.letter_date > week or not letter.fei:
            continue
        ei = estab_index.get(letter.fei)
        if ei is not None:
            letters_by_estab[ei].append(letter)

    # --- drug features ------------------------------------------------------
    Xd = np.zeros((n_d, len(FEATURE_NAMES_DRUG)), dtype=np.float32)
    classes = ("Class I", "Class II", "Class III")
    for key, products in drugs.items():
        i = drug_index[key]
        drug_labelers = {p.labeler_code for p in products}
        drug_estabs: set[str] = set()
        for product in products:
            drug_estabs.update(raw.product_estabs.get(product.ndc, set()))
        listings = defaultdict(int)
        for product in products:
            listings[product.labeler_code] += 1
        total = sum(listings.values()) or 1
        hhi = sum((count / total) ** 2 for count in listings.values())

        outside = sum(
            1 for fei in drug_estabs
            if raw.establishments.get(fei, {}).get("country", "US").upper() not in {"US", "USA", "UNITED STATES"}
        )
        share_outside = outside / len(drug_estabs) if drug_estabs else 0.0
        sterile = 1.0 if any(
            p.route.upper() in {"INTRAVENOUS", "IV", "INJECTION", "SUBCUTANEOUS"}
            and "INJECT" in p.dosage_form.upper() or "INJECTION" in p.dosage_form.upper()
            for p in products
        ) else 0.0

        Xd[i, 0] = len(drug_labelers)
        Xd[i, 1] = len(drug_estabs)
        Xd[i, 2] = hhi
        Xd[i, 3] = share_outside
        Xd[i, 4] = sterile

        recalls = drug_recalls.get(i, [])
        col = 5
        for window in TRAILING_WINDOWS:
            for cls in classes:
                Xd[i, col] = sum(
                    1 for date, c in recalls
                    if c == cls and 0 <= _weeks_between(date, week) <= window
                )
                col += 1
        recent = [ (week - date).days / 7 for date, _ in recalls ]
        Xd[i, 14] = min(min(recent), 200.0) if recent else 200.0

        # letters propagate through establishments (two hops)
        drug_letters: list[WarningLetter] = []
        for fei in drug_estabs:
            ei = estab_index.get(fei)
            if ei is not None:
                drug_letters.extend(letters_by_estab.get(ei, []))
        for offset, window in enumerate(TRAILING_WINDOWS):
            Xd[i, 15 + offset] = sum(
                1 for letter in drug_letters
                if 0 <= _weeks_between(letter.letter_date, week) <= window
            )
        letter_ages = [(week - letter.letter_date).days / 7 for letter in drug_letters]
        Xd[i, 18] = min(min(letter_ages), 200.0) if letter_ages else 200.0
        Xd[i, 19] = max((len(letter.cgmp_categories) for letter in drug_letters), default=0)
        Xd[i, 20] = 1.0 if any(letter.import_alert for letter in drug_letters) else 0.0

    # --- labeler features ---------------------------------------------------
    Xl = np.zeros((n_l, len(FEATURE_NAMES_LABELER)), dtype=np.float32)
    for name, i in labeler_index.items():
        recalls = labeler_recalls.get(i, [])
        col = 0
        for cls in classes:
            Xl[i, col] = sum(1 for date, c in recalls
                             if c == cls and 0 <= _weeks_between(date, week) <= 26)
            col += 1
        Xl[i, 3] = 0.0  # letters attach to establishments, not labelers
        ages = [(week - date).days / 7 for date, _ in recalls]
        Xl[i, 4] = min(min(ages), 200.0) if ages else 200.0
        Xl[i, 5] = 200.0
        Xl[i, 6] = sum(1 for p in active if p.labeler_code == name)

    # --- establishment features --------------------------------------------
    Xe = np.zeros((n_e, len(FEATURE_NAMES_ESTAB)), dtype=np.float32)
    for fei, i in estab_index.items():
        letters = letters_by_estab.get(i, [])
        Xe[i, 0] = 0.0
        Xe[i, 1] = sum(1 for letter in letters
                       if 0 <= _weeks_between(letter.letter_date, week) <= 26)
        ages = [(week - letter.letter_date).days / 7 for letter in letters]
        Xe[i, 2] = min(min(ages), 200.0) if ages else 200.0
        country = raw.establishments.get(fei, {}).get("country", "US").upper()
        Xe[i, 3] = 1.0 if country in {"US", "USA", "UNITED STATES"} else 0.0

    # --- edges --------------------------------------------------------------
    edges: dict[str, list[tuple[int, int]]] = {name: [] for name in RELATIONS}
    for key, products in drugs.items():
        di = drug_index[key]
        for product in products:
            li = labeler_index[product.labeler_code]
            edges["markets"].append((di, offset_l + li))
            for fei in raw.product_estabs.get(product.ndc, set()):
                ei = estab_index.get(fei)
                if ei is not None:
                    edges["made_at"].append((di, offset_e + ei))
                    edges["manufactured_for"].append((offset_l + li, offset_e + ei))

    # substitutes: same ingredient set, different form or route
    by_ingredients: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for key in drug_keys:
        ingredients, form, route = (key.split("|") + ["", ""])[:3]
        by_ingredients[tuple(ingredients.split("+"))].append(key)
    for keys in by_ingredients.values():
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                _, form_a, route_a = (keys[a].split("|") + ["", ""])[:3]
                _, form_b, route_b = (keys[b].split("|") + ["", ""])[:3]
                if (form_a, route_a) != (form_b, route_b):
                    u, v = drug_index[keys[a]], drug_index[keys[b]]
                    edges["substitutes"].append((u, v))
                    edges["substitutes"].append((v, u))

    frozen: dict[str, np.ndarray] = {}
    for name, pairs in edges.items():
        arr = np.array(pairs, dtype=np.int64).reshape(-1, 2).T if pairs else np.zeros((2, 0), dtype=np.int64)
        frozen[name] = arr
        reverse = f"{name}_rev"
        if reverse in RELATIONS:
            frozen[reverse] = arr[::-1].copy()

    return Snapshot(
        week=week,
        drug_keys=drug_keys,
        labeler_ids=labelers,
        estab_ids=estab_ids,
        Xd=Xd,
        Xl=Xl,
        Xe=Xe,
        edges=frozen,
        split=split_of(week),
        embargoed=embargo_excludes(week),
    )


def build_snapshots(raw: RawData, start: dt.date, end: dt.date) -> list[Snapshot]:
    """Build every weekly snapshot in the window, in order."""
    return [build_snapshot(raw, week) for week in iter_weeks(start, end)]


@dataclass
class Labels:
    y: np.ndarray            # (n_drugs, n_horizons), 1 if first posting within horizon
    at_risk: np.ndarray      # (n_drugs,), True if never listed as of this week
    first_posting: np.ndarray  # (n_drugs,) week index or -1


def label_snapshot(
    snapshot: Snapshot, shortages: list[ShortagePosting], horizon_weeks: tuple[int, ...] = HORIZONS_WEEKS
) -> Labels:
    """First-onset labels. Planned discontinuations are excluded entirely."""
    first: dict[str, dt.date] = {}
    for posting in shortages:
        if posting.planned_discontinuation:
            continue
        key = drug_key(posting.drug_key[0], posting.drug_key[1], posting.drug_key[2])
        if key not in first or posting.posting_date < first[key]:
            first[key] = posting.posting_date

    n = len(snapshot.drug_keys)
    y = np.zeros((n, len(horizon_weeks)), dtype=np.float32)
    at_risk = np.zeros(n, dtype=bool)
    first_arr = np.full(n, -1, dtype=np.int64)

    for i, key in enumerate(snapshot.drug_keys):
        posting = first.get(key)
        if posting is None:
            at_risk[i] = True  # never listed by the end of the study window
            continue
        if posting <= snapshot.week:
            first_arr[i] = 0  # already listed: not at risk, not a candidate
            continue
        at_risk[i] = True
        delta = (posting - snapshot.week).days / 7.0
        first_arr[i] = int(round(delta))
        for j, horizon in enumerate(horizon_weeks):
            if delta <= horizon:
                y[i, j] = 1.0

    return Labels(y=y, at_risk=at_risk, first_posting=first_arr)
