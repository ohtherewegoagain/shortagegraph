"""Synthetic fixture generator.

NOT DATA. This exists so the pipeline can be executed end to end in the test
suite: the code path, the temporal split, the hazard head and the metrics all
run, and the graph carries a planted mechanism (a facility event raising risk
for products made there) that a working model should be able to exploit.

Numbers produced from this fixture must never appear in the paper. The real
run reads FDA sources through `shortagegraph.ingest`.

The generator writes `data/fixtures/synthetic.json` so a fixture run is
reproducible and inspectable, and returns a `RawData` object directly.
"""

from __future__ import annotations

import datetime as dt
import json
import random

from .graph import RawData, drug_key
from .ingest import NdcProduct, RecallReport, ShortagePosting, WarningLetter

INGREDIENTS = [
    "VANCOMYCIN", "PIPERACILLIN", "MEROPENEM", "CISPLATIN", "DOXORUBICIN",
    "INSULIN ASPART", "PROPOFOL", "KETAMINE", "FENTANYL", "MORPHINE",
    "AMIKACIN", "CEFEPIME", "DEXAMETHASONE", "ONDANSETRON", "HEPARIN",
    "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "LEVOTHYROXINE", "METFORMIN",
    "AMOXICILLIN",
]
FORMS = ["INJECTION", "TABLET", "CAPSULE", "SOLUTION"]
ROUTES = ["INTRAVENOUS", "ORAL", "SUBCUTANEOUS", "TOPICAL"]


def make_raw(
    *,
    seed: int = 7,
    n_labelers: int = 14,
    n_estabs: int = 18,
    n_drugs: int = 60,
    start: dt.date = dt.date(2018, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
    onset_fraction: float = 0.35,
) -> RawData:
    rng = random.Random(seed)

    labelers = [f"LAB_{i:03d}" for i in range(n_labelers)]
    estabs = {f"FEI{i:07d}": {"name": f"PLANT {i}", "country": "US" if i % 5 else "IN"} for i in range(n_estabs)}
    estab_ids = list(estabs)

    products: list[NdcProduct] = []
    product_estabs: dict[str, set[str]] = {}

    for d in range(n_drugs):
        ingredients = frozenset(rng.sample(INGREDIENTS, k=rng.choice([1, 1, 2])))
        form = rng.choice(FORMS)
        route = rng.choice(ROUTES)
        # single-source drugs are a minority, as in the real network
        n_makers = 1 if rng.random() < 0.4 else rng.choice([2, 3])
        makers = rng.sample(labelers, k=n_makers)
        for m_idx, labeler in enumerate(makers):
            ndc = f"{hash((d, m_idx, seed)) % 100000:05d}-{d:04d}-{m_idx}"
            feis = set(rng.sample(estab_ids, k=rng.choice([1, 1, 2])))
            product_estabs[ndc] = feis
            products.append(
                NdcProduct(
                    ndc=ndc,
                    labeler_code=labeler,
                    labeler_name=labeler,
                    ingredient_set=ingredients,
                    dosage_form=form,
                    route=route,
                    marketing_start=start - dt.timedelta(days=rng.randint(0, 2000)),
                    marketing_end=None if rng.random() < 0.85 else end + dt.timedelta(days=400),
                )
            )

    # A subset of establishments get a warning letter; products made there are
    # the ones that go short a few weeks later. This is the mechanism of Fig. 1.
    troubled = rng.sample(estab_ids, k=max(1, n_estabs // 3))
    letters: list[WarningLetter] = []
    for fei in troubled:
        year = rng.randint(2019, 2025)
        letter_date = dt.date(year, rng.randint(1, 12), rng.randint(1, 27))
        letters.append(
            WarningLetter(
                letter_date=letter_date,
                firm=estabs[fei]["name"],
                fei=fei,
                cgmp_categories=frozenset(
                    rng.sample(["sterility", "contamination", "data_integrity"], k=rng.randint(1, 2))
                ),
                import_alert=rng.random() < 0.2,
                raw_text=f"synthetic warning letter for {fei}",
            )
        )

    # Shortages: drugs made at troubled plants, posted 4-16 weeks after the
    # letter. Some untroubled drugs also go short, without warning.
    shortages: list[ShortagePosting] = []
    letter_by_fei = {l.fei: l for l in letters}
    seen: set[str] = set()
    for product in products:
        key = drug_key(product.ingredient_set, product.dosage_form, product.route)
        if key in seen:
            continue
        feis = product_estabs[product.ndc]
        trigger = next((letter_by_fei[f] for f in feis if f in letter_by_fei), None)
        if trigger and rng.random() < onset_fraction:
            posting = trigger.letter_date + dt.timedelta(weeks=rng.randint(4, 16))
            if posting <= end:
                shortages.append(
                    ShortagePosting(
                        drug_key=(key.split("|")[0], key.split("|")[1], key.split("|")[2]),
                        posting_date=posting,
                        product_name=key,
                    )
                )
                seen.add(key)
        elif rng.random() < 0.05:
            posting = start + dt.timedelta(weeks=rng.randint(52, 380))
            if posting <= end:
                shortages.append(
                    ShortagePosting(
                        drug_key=(key.split("|")[0], key.split("|")[1], key.split("|")[2]),
                        posting_date=posting,
                        product_name=key,
                    )
                )
                seen.add(key)

    # Recalls, mostly after letters, attached to labeler NDCs.
    recalls: list[RecallReport] = []
    for product in products:
        if rng.random() < 0.12:
            base = letter_by_fei.get(next(iter(product_estabs[product.ndc]), ""), None)
            when = (base.letter_date if base else start) + dt.timedelta(weeks=rng.randint(1, 60))
            if when > end:
                when = end
            recalls.append(
                RecallReport(
                    report_date=when,
                    classification=rng.choice(["Class I", "Class II", "Class II", "Class III"]),
                    firm=product.labeler_name,
                    product_description=f"{product.dosage_form} {product.route}",
                    ndc_codes=frozenset({product.ndc[:9]}),
                )
            )

    raw = RawData(
        products=products,
        establishments=estabs,
        product_estabs=product_estabs,
        recalls=recalls,
        letters=letters,
        shortages=shortages,
    )
    return raw


def write_fixture(raw: RawData, path) -> None:
    """Serialise a fixture for inspection (not read back by the pipeline)."""
    payload = {
        "products": [
            {
                "ndc": p.ndc,
                "labeler": p.labeler_code,
                "ingredients": sorted(p.ingredient_set),
                "form": p.dosage_form,
                "route": p.route,
            }
            for p in raw.products
        ],
        "establishments": raw.establishments,
        "letters": [
            {"date": l.letter_date.isoformat(), "fei": l.fei, "firm": l.firm} for l in raw.letters
        ],
        "shortages": [
            {"key": "|".join(s.drug_key), "date": s.posting_date.isoformat()} for s in raw.shortages
        ],
        "recalls": [
            {"date": r.report_date.isoformat(), "class": r.classification} for r in raw.recalls
        ],
        "note": "SYNTHETIC FIXTURE - for pipeline tests only, never for paper results",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
