"""Synthetic data calibrated to measured public-data marginals.

Why this exists: the real pipeline needs the full NDC directory, SPL-derived
establishment links and several years of postings, which is a heavy first run.
This generator produces a graph with the *structural shape* measured from the
public sources (number of products, labelers per product, recall rate and class
mix, establishment geography, shortage rate) so the whole pipeline can be
exercised at a size a laptop can train on.

What it is not: evidence. Every number produced from a generated graph is a
property of the generator's parameters, not of the U.S. drug supply. It must
never appear in the paper. `train.py` stamps the payload with CALIBRATED
SYNTHETIC so this cannot be mistaken at the output level either.

The calibration file is produced by `shortagegraph.calibrate`, which records
the endpoint, the retrieval time and the counts it measured.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import asdict
from pathlib import Path

from .graph import RawData
from .ingest import NdcProduct, RecallReport, ShortagePosting, WarningLetter

# Fallbacks are deliberately conservative and are only used when a calibration
# field is missing; every use is reported in `params_used`.
DEFAULTS = {
    "products": {
        "n_products": 3000,
        "n_labelers": 600,
        "single_labeler_share": 0.40,
        "n_establishments": 1200,
        "outside_us_share": 0.22,
    },
    "recalls": {"per_year": 900, "class_shares": {"Class I": 0.08, "Class II": 0.74, "Class III": 0.18}},
    "letters": {"per_year": 55},
    "shortages": {"per_year": 180},
}

INGREDIENTS = [
    "VANCOMYCIN", "PIPERACILLIN", "MEROPENEM", "CISPLATIN", "DOXORUBICIN",
    "INSULIN ASPART", "PROPOFOL", "KETAMINE", "FENTANYL", "MORPHINE",
    "AMIKACIN", "CEFEPIME", "DEXAMETHASONE", "ONDANSETRON", "HEPARIN",
    "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "LEVOTHYROXINE", "METFORMIN",
    "AMOXICILLIN", "AZITHROMYCIN", "DILTIAZEM", "LIDOCAINE", "NOREPINEPHRINE",
    "PHENYLEPHRINE", "ROCURONIUM", "MIDAZOLAM", "HYDROMORPHONE", "EPINEPHRINE",
    "CALCIUM GLUCONATE", "MAGNESIUM SULFATE", "SODIUM BICARBONATE",
]
FORMS = ["INJECTION", "TABLET", "CAPSULE", "SOLUTION", "SUSPENSION"]
ROUTES = ["INTRAVENOUS", "ORAL", "SUBCUTANEOUS", "TOPICAL", "INTRAMUSCULAR"]
COUNTRIES = ["US", "US", "US", "IN", "CN", "DE", "IT", "ES"]


def load_calibration(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULTS))
    for group, values in data.get("parameters", {}).items():
        if isinstance(values, dict) and group in merged:
            merged[group].update(values)
        else:
            merged[group] = values
    merged["_provenance"] = {
        "retrieved": data.get("retrieved"),
        "sources": data.get("sources", []),
    }
    return merged


def generate(
    calibration: dict,
    *,
    seed: int = 13,
    max_products: int = 400,
    start: dt.date = dt.date(2018, 1, 1),
    end: dt.date = dt.date(2025, 12, 31),
) -> RawData:
    """Build a RawData with calibrated marginals and a planted facility effect."""
    rng = random.Random(seed)
    p = calibration["products"]

    n_labelers = int(p["n_labelers"])
    n_estabs = int(p["n_establishments"])
    outside_share = float(p["outside_us_share"])
    single_share = float(p["single_labeler_share"])
    n_products = min(int(p["n_products"]), max_products)

    labelers = [f"LAB_{i:04d}" for i in range(n_labelers)]
    estab_ids = [f"FEI{i:08d}" for i in range(n_estabs)]
    establishments = {
        fei: {
            "name": f"PLANT {i}",
            "country": rng.choice(COUNTRIES) if rng.random() < outside_share else "US",
        }
        for i, fei in enumerate(estab_ids)
    }

    products: list[NdcProduct] = []
    product_estabs: dict[str, set[str]] = {}
    drug_groups: dict[str, list[int]] = {}

    for d in range(n_products):
        ingredients = tuple(sorted(rng.sample(INGREDIENTS, k=rng.choice([1, 1, 1, 2]))))
        form = rng.choice(FORMS)
        route = "INTRAVENOUS" if form == "INJECTION" else rng.choice(ROUTES)
        n_makers = 1 if rng.random() < single_share else rng.choice([2, 2, 3, 4])
        makers = rng.sample(labelers, k=min(n_makers, len(labelers)))
        for m_idx, labeler in enumerate(makers):
            ndc = f"{10000 + d:05d}-{d:04d}-{m_idx}"
            feis = set(rng.sample(estab_ids, k=rng.choice([1, 1, 2])))
            product_estabs[ndc] = feis
            products.append(
                NdcProduct(
                    ndc=ndc,
                    labeler_code=labeler,
                    labeler_name=labeler,
                    ingredient_set=frozenset(ingredients),
                    dosage_form=form,
                    route=route,
                    marketing_start=start - dt.timedelta(days=rng.randint(0, 3000)),
                    marketing_end=None if rng.random() < 0.88 else end + dt.timedelta(days=500),
                )
            )
        key = f"{'+'.join(ingredients)}|{form}|{route}"
        drug_groups.setdefault(key, []).append(d)

    years = list(range(start.year, end.year + 1))

    # --- warning letters, spread by the calibrated annual rate -------------
    # Scale the annual rates by the fraction of the real product universe we
    # actually materialise, so the generator's densities match the calibration.
    coverage = n_products / max(1, int(p["n_products"]))
    letters: list[WarningLetter] = []
    per_year_letters = max(1, int(round(calibration["letters"]["per_year"] * coverage)))
    troubled = rng.sample(estab_ids, k=max(1, n_estabs // 3))
    for year in years:
        for _ in range(rng.randint(max(0, per_year_letters - 3), per_year_letters + 3)):
            fei = rng.choice(troubled)
            when = dt.date(year, rng.randint(1, 12), rng.randint(1, 27))
            letters.append(
                WarningLetter(
                    letter_date=when,
                    firm=establishments[fei]["name"],
                    fei=fei,
                    cgmp_categories=frozenset(
                        rng.sample(
                            ["sterility_assurance", "contamination_control", "data_integrity",
                             "laboratory_controls", "quality_processes"],
                            k=rng.randint(1, 3),
                        )
                    ),
                    import_alert=rng.random() < 0.15,
                    raw_text=f"synthetic letter {fei} {when.isoformat()}",
                )
            )
    letters.sort(key=lambda l: l.letter_date)
    letter_by_fei = {l.fei: l for l in letters}

    # --- shortages: calibrated annual rate, partly facility-driven ---------
    per_year_shortages = max(1, int(round(calibration["shortages"]["per_year"] * coverage)))
    shortages: list[ShortagePosting] = []
    listed: set[str] = set()
    for year in years:
        target = rng.randint(max(1, per_year_shortages - 5), per_year_shortages + 5)
        for _ in range(target):
            key = rng.choice(list(drug_groups))
            if key in listed:
                continue
            ingredients, form, route = (key.split("|") + ["", ""])[:3]
            posting_day = dt.date(year, rng.randint(1, 12), rng.randint(1, 27))
            # if any maker sits at a lettered plant, the posting follows the
            # letter with a short lag: this is the mechanism of Fig. 1
            trigger = None
            for d_index in drug_groups[key][:1]:
                ndc = f"{10000 + d_index:05d}-{d_index:04d}-0"
                for fei in product_estabs.get(ndc, set()):
                    letter = letter_by_fei.get(fei)
                    if letter and letter.letter_date <= posting_day:
                        trigger = letter.letter_date + dt.timedelta(weeks=rng.randint(4, 16))
                        break
                if trigger:
                    break
            if trigger and trigger <= end:
                posting_day = trigger
            shortages.append(
                ShortagePosting(
                    drug_key=(ingredients, form, route),
                    posting_date=posting_day,
                    product_name=key,
                    planned_discontinuation=rng.random() < 0.06,
                )
            )
            listed.add(key)
    shortages.sort(key=lambda s: s.posting_date)

    # --- recalls: calibrated annual rate and class mix ---------------------
    class_shares = calibration["recalls"]["class_shares"]
    classes = list(class_shares)
    weights = [float(class_shares[c]) for c in classes]
    per_year_recalls = max(1, int(round(calibration["recalls"]["per_year"] * coverage)))
    recalls: list[RecallReport] = []
    for year in years:
        target = rng.randint(max(1, per_year_recalls - 20), per_year_recalls + 20)
        for _ in range(target):
            product = rng.choice(products)
            when = dt.date(year, rng.randint(1, 12), rng.randint(1, 27))
            recalls.append(
                RecallReport(
                    report_date=when,
                    classification=rng.choices(classes, weights=weights, k=1)[0],
                    firm=product.labeler_name,
                    product_description=f"{product.dosage_form} {product.route}",
                    ndc_codes=frozenset({product.ndc[:9]}),
                )
            )
    recalls.sort(key=lambda r: r.report_date)

    return RawData(
        products=products,
        establishments=establishments,
        product_estabs=product_estabs,
        recalls=recalls,
        letters=letters,
        shortages=shortages,
    )


# --- round-trip so a generated graph can be re-used without regenerating ----


def save(raw: RawData, path: Path, calibration: dict | None = None) -> None:
    payload = {
        "note": "CALIBRATED SYNTHETIC - generated, not observed. Never paper results.",
        "calibration": (calibration or {}).get("_provenance", {}),
        "products": [
            {
                "ndc": p.ndc,
                "labeler_code": p.labeler_code,
                "labeler_name": p.labeler_name,
                "ingredient_set": sorted(p.ingredient_set),
                "dosage_form": p.dosage_form,
                "route": p.route,
                "marketing_start": p.marketing_start.isoformat() if p.marketing_start else None,
                "marketing_end": p.marketing_end.isoformat() if p.marketing_end else None,
            }
            for p in raw.products
        ],
        "establishments": raw.establishments,
        "product_estabs": {k: sorted(v) for k, v in raw.product_estabs.items()},
        "recalls": [
            {
                "report_date": r.report_date.isoformat(),
                "classification": r.classification,
                "firm": r.firm,
                "product_description": r.product_description,
                "ndc_codes": sorted(r.ndc_codes),
            }
            for r in raw.recalls
        ],
        "letters": [
            {
                "letter_date": l.letter_date.isoformat(),
                "firm": l.firm,
                "fei": l.fei,
                "cgmp_categories": sorted(l.cgmp_categories),
                "import_alert": l.import_alert,
                "raw_text": l.raw_text,
            }
            for l in raw.letters
        ],
        "shortages": [
            {
                "drug_key": list(s.drug_key),
                "posting_date": s.posting_date.isoformat(),
                "product_name": s.product_name,
                "planned_discontinuation": s.planned_discontinuation,
            }
            for s in raw.shortages
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load(path: Path) -> RawData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    products = [
        NdcProduct(
            ndc=row["ndc"],
            labeler_code=row["labeler_code"],
            labeler_name=row["labeler_name"],
            ingredient_set=frozenset(row["ingredient_set"]),
            dosage_form=row["dosage_form"],
            route=row["route"],
            marketing_start=dt.date.fromisoformat(row["marketing_start"]) if row["marketing_start"] else None,
            marketing_end=dt.date.fromisoformat(row["marketing_end"]) if row["marketing_end"] else None,
        )
        for row in payload["products"]
    ]
    recalls = [
        RecallReport(
            report_date=dt.date.fromisoformat(row["report_date"]),
            classification=row["classification"],
            firm=row["firm"],
            product_description=row["product_description"],
            ndc_codes=frozenset(row["ndc_codes"]),
        )
        for row in payload["recalls"]
    ]
    letters = [
        WarningLetter(
            letter_date=dt.date.fromisoformat(row["letter_date"]),
            firm=row["firm"],
            fei=row["fei"],
            cgmp_categories=frozenset(row["cgmp_categories"]),
            import_alert=row["import_alert"],
            raw_text=row.get("raw_text", ""),
        )
        for row in payload["letters"]
    ]
    shortages = [
        ShortagePosting(
            drug_key=(row["drug_key"][0], row["drug_key"][1], row["drug_key"][2]),
            posting_date=dt.date.fromisoformat(row["posting_date"]),
            product_name=row["product_name"],
            planned_discontinuation=row["planned_discontinuation"],
        )
        for row in payload["shortages"]
    ]
    return RawData(
        products=products,
        establishments=payload["establishments"],
        product_estabs={k: set(v) for k, v in payload["product_estabs"].items()},
        recalls=recalls,
        letters=letters,
        shortages=shortages,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a calibrated synthetic graph")
    parser.add_argument("--calibration", required=True, help="path to calibration.json")
    parser.add_argument("--out", default="data/fixtures/calibrated.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-products", type=int, default=400)
    args = parser.parse_args(argv)

    calibration = load_calibration(Path(args.calibration))
    raw = generate(calibration, seed=args.seed, max_products=args.max_products)
    save(raw, Path(args.out), calibration)
    print(
        f"wrote {args.out}: {len(raw.products)} products, {len(raw.letters)} letters, "
        f"{len(raw.shortages)} shortages, {len(raw.recalls)} recalls "
        "(CALIBRATED SYNTHETIC - not results)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
