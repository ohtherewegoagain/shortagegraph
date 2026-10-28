# ShortageGraph — EMBC 2027 submission

Manuscript in `embc/`; analysis code in `shortagegraph/`.

```
embc/
  main.tex         # IEEEtran[conference]: 4 pages content + page 5 references
  main.pdf         # compiled draft
  PLACEHOLDERS.md  # the few remaining blanks, and pre-upload checks
shortagegraph/     # the released code behind every number in the paper
archive/           # superseded drafts
```

This directory is the complete public repository, ready to upload as-is.

## What the paper is now

The retrospective forecasting design could not be **trained**, because its labels
cannot be certified from any reachable public source (Section IV of the paper,
and `shortagegraph/labels.py`). Rather than submit a model trained on corrupt
negatives, the paper reports what the data can support and keeps the system
specification as a reproducible benchmark:

1. **A measured network** — 114,018 NDC products aggregated over 96 monthly
   snapshots: 10,537 drug nodes, 4,391 labelers, 88,071 edges, **68.0%**
   single-labeler.
2. **A cross-sectional association study** at one reference date (22 Oct 2026):
   293 drugs on the FDA shortage list vs 14,313 marketed drugs not on it.
   Same-product recall in the prior 24 months: 17.1% vs 2.5% (OR 8.07,
   95% CI 5.85–11.13); **3.64** after stratifying on product count; 4.29
   in injectables.
3. **A label-provenance audit** — 1,142 current vs **19 resolved** records, so
   first-onset labels are survivorship-biased; the released gate refuses to
   train on years it cannot certify.
4. **The forecasting system, specified but not scored** — relational message
   passing, a monotone multi-horizon hazard head, occlusion attribution, four
   baselines and the full evaluation protocol. Implemented and unit-tested, with
   no performance reported: a number computed on uncertified labels would
   describe the bias, not the model.

Every number in the paper is produced by code in this repo. Nothing is
synthetic.

## Run it

```bash
export PYTHONPATH=./.pylibs

python3 -m pytest tests -q                                   # 20 tests
python3 -m shortagegraph.realdata                            # Table I (network)
python3 -m shortagegraph.casecontrol --window-months 24      # Tables II-III
python3 -m shortagegraph.labels --audit                      # Section V audit
```

`casecontrol` downloads and caches the enforcement reports it needs on first
run (`data/raw/recalls_<start>_<end>.json`); reruns are offline and
deterministic. Outputs land in `data/runs/casecontrol/`.

## EMBC 2027 rules this draft follows

Verified against `embc.embs.org/2027/author-guidelines` on 2026-10-28:

- **4+1 pages**: pages 1–4 technical content; page 5 acknowledgments and
  references only, enforced with `\clearpage`.
- **≤20 references**: this draft has 12, all cited.
- **Stock IEEEtran**, no margin/font tweaks — non-compliant layout is a desk
  reject.
- **AI disclosure**: AI-generated content must be named in the Acknowledgment
  with sections and scope. Undeclared use is grounds for rejection, as are
  fabricated citations or synthetic results.

The CFP lists **28 February 2027** for full papers; confirm when the portal
opens in December 2026.

## Build

```bash
cd embc && tectonic main.tex
```

## The data blocker, stated plainly

The openFDA shortage mirror is a live snapshot, not an archive: **1,142 current
entries and only 19 resolved**. Per-year first-posting counts across the
intended training window are 13 (2018), 4 (2019), 17 (2020), 11 (2021), 18
(2022), against the hundreds the FDA recorded annually. A missing posting makes
an already-short drug look never-listed, which corrupts training instead of
merely thinning it.

The escape routes are closed: the Wayback CDX index has no usable snapshots of
the legacy `dsp_ActiveIngredientList.cfm` page, and `datadashboard.fda.gov`
returns 403 to automated clients. A complete history needs an FDA archival
export, a commercial shortage dataset, or a partner health system's records.
`shortagegraph/archive_import.py` loads such a history the moment one exists,
and refuses to write a file that fails the same certification.

## Limits worth remembering

- A single cross-section estimates association at one date — not incidence, not
  causation, not forecast skill.
- Cases are ascertained through NDC codes, so large products are easier to
  capture; that is why the concentration result (OR 0.14) is reported as an
  artefact and not as a finding.
- The broad "any-labeler recall" exposure is close to a measure of firm size;
  the paper relies on the same-product definition.
