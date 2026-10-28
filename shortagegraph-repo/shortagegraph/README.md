# ShortageGraph pipeline

Code for the paper's Sections II–V: weekly supply-network snapshots, risk
features, first-onset labels, the R-GCN with a monotone hazard head, four
baselines, and the evaluation protocol.

## Status, plainly

| Part | State |
|---|---|
| Graph construction (II-A) | implemented, dated availability, reverse edges |
| Risk features (II-B) | implemented, trailing 13/26/52-week windows |
| Warning-letter LLM extraction + audit | implemented; needs `OPENAI_API_KEY` to run live |
| Labels and temporal split (II-C) | implemented, 12-week embargo |
| R-GCN + monotone head (III-A/B) | implemented in PyTorch |
| Baselines (III-C) | sole-source rule, logistic, gradient boosting, edge-free MLP |
| Evaluation (IV-A) | AUPRC/AUROC/P@25/Brier, drug-level bootstrap, lead time |
| Table I / Table II numbers | **not produced.** Needs the real FDA inputs |

Nothing in this repository has been run on real FDA data yet. The only
executable path today is the synthetic fixture, which exists to prove the code
runs and must never supply paper numbers.

## Layout

```
shortagegraph/
  config.py     study period, horizons, splits, embargo, source URLs
  ingest.py     downloaders and loaders for the five FDA sources
  graph.py      snapshot builder, features, labels
  extract.py    warning-letter extraction (LLM) + hand-audit harness
  model.py      R-GCN, monotone hazard head, edge-free MLP, weighted loss
  baselines.py  rule, logistic regression, gradient boosting
  train.py      training loop, evaluation, Table I/II output
  fixture.py    synthetic generator (tests only)
tests/
  test_smoke.py unit + end-to-end tests
```

## Run the tests

```bash
python3 -m pip install --target ./.pylibs -r requirements.txt
PYTHONPATH=./.pylibs python3 -m pytest tests -q
```

Or a single fixture run:

```bash
PYTHONPATH=./.pylibs python3 -m shortagegraph.train --fixture --epochs 20 --seeds 2 --out data/runs/fixture
```

This writes `results.json` and `results.md` with the same columns as Table II.

## Run on real data

Real runs need five inputs under `data/raw/`. Nothing is bundled, because each
is large or licence-restricted.

1. **NDC directory** — `python -c "from shortagegraph.ingest import *; fetch_ndc_directory(Paths().ensure())"`,
   or drop the unzipped `ndctext.zip` at `data/raw/ndc/`.
2. **Product → establishment links** — `data/raw/product_estabs.jsonl`, one
   object per NDC: `{"ndc": "...", "feis": ["..."]}`, produced by parsing the
   establishments named in each product's SPL in DailyMed. This is the step
   that turns labeling text into `made at` edges; the paper's entity resolution
   claims rest on it.
3. **Establishment index** — `data/raw/establishments.jsonl`:
   `{"fei": "...", "name": "...", "country": "US"}`.
4. **Recalls** — fetched from openFDA, or cached at `data/raw/recalls.jsonl`.
5. **Warning letters** — `data/raw/letters.jsonl` in the format
   `extract.py` writes.
6. **Shortages** — `data/raw/shortages.csv` with columns `drug_key`,
   `posting_date`, `product_name`, `planned_discontinuation`, where `drug_key`
   is `INGREDIENTS|FORM|ROUTE` matching `graph.drug_key`.

Then:

```bash
PYTHONPATH=./.pylibs python3 -m shortagegraph.train --from-raw data/raw --epochs 40 --seeds 5 --out data/runs/real
```

`ingest.load_raw_from_dir` reads that layout into `RawData`. It fails loudly if
the NDC→FEI mapping or the recalls file is absent, because a graph without
those edges cannot support the paper's two-hop claim.

## Calibration and the synthetic graph

The public endpoints are reachable, so the generator's parameters are measured
rather than invented:

```bash
python3 -m shortagegraph.calibrate --out data/raw/calibration.json   # 10.8 MB NDC zip + openFDA
python3 -m shortagegraph.synthesize --calibration data/raw/calibration.json \
    --out data/fixtures/calibrated.json --max-products 400
python3 -m shortagegraph.train --synthetic data/fixtures/calibrated.json \
    --epochs 8 --seeds 1 --out data/runs/calibrated
```

`calibrate.py` writes what it measured (NDC product/labeler counts, the
single-labeler share, per-year shortage and recall totals, recall class mix)
and separately what it could not measure and had to assume (establishment
count, share of establishments outside the U.S., warning letters per year — the
FDA Data Dashboard returns 403 to automated clients from this host). Both lists
are in `calibration.json` under `measured` and `assumed`.

**A graph generated this way is not an observation.** It reproduces the shape of
the network, not its behaviour, so every metric computed from it is a property
of the generator. It exists to prove the pipeline runs at a realistic size. It
cannot fill Table II, and `train.py` stamps the payload `CALIBRATED SYNTHETIC`
so the output cannot be mistaken for a result later.

## Labels and the completeness guard

The label definition (first posting week) is only sound if the posting history is
complete for the window being labelled. Audit it before trusting any run:

```bash
python3 -m shortagegraph.labels --audit     # writes data/raw/label_coverage.json
```

Current state of the public source: **not certified.** The endpoint serves the
live database (1,142 `Current`, 19 `Resolved`), so resolved shortages are absent
and per-year onsets collapse to 4–31 where the FDA recorded hundreds. Years below
`min_year_onsets` are rejected, because a thin year means lost records rather than
a quiet year.

`train.py --from-raw` calls `certify()` and exits rather than training on an
uncertified history; `--force-uncertified-labels` exists only for debugging. A
graph of the wrong shape is obvious, but corrupted labels produce ordinary-looking
numbers, which is why this is a hard stop instead of a warning.

## What this code will and will not settle

It will produce every number in Table II from real inputs, and the dataset
counts in Table I. It will not:

- confirm the paper's claims. If the GNN does not beat gradient boosting, the
  code will say so and the paper text must change.
- validate the LLM extraction. That needs the hand audit: `audit_sample` then
  `audit_score`, on a real sample of letters.
- run at production scale. Snapshot building is a Python loop per week; on the
  full NDC directory expect it to be slow, and vectorise before a large sweep.

## Modules added with the cross-sectional reframe

| Module | What it produces |
|---|---|
| `casecontrol.py` | Tables II–III of the paper: odds ratios, Fisher exact tests, Mantel–Haenszel pooling, product-count stratification, injectable subgroup, 12/24/36-month sensitivity |
| `realdata.py` | Table I: network statistics measured from the NDC directory |
| `labels.py` | Section IV: the posting-history coverage audit and its certification gate |
| `archive_import.py` | Loads a complete shortage history from an external source, then applies the same gate |

`casecontrol.py` is the only module that needs the network at run time; it
caches the enforcement reports it downloads, so a rerun is offline and
deterministic. It reports the 2x2 table behind every estimate, which is what
makes the numbers in the paper checkable rather than merely stated.
