# Submission readiness: evidence register and checklist

Built from the internal documents in this Drive folder (`Platform_Overview`,
`Capabilities_and_Modules`, `LRRK2_Case_Study`, `Lonnie_Technical_Packet`).
This file is the internal companion to `main.tex`. **Never submit this file** —
it is not anonymized and it names source documents.

---

## 1. Blocking problems, in priority order

### 1.1 There is no experimental evidence of any kind

Every source document says the same thing: `Platform_Overview` (2026-10-16),
"All results in-silico. No wet-lab validation yet"; `Capabilities_and_Modules`,
"No wet-lab work in house"; `LRRK2_Case_Study`, "No compound from this campaign
has been synthesized or tested."

Consequence: the paper cannot claim discovery, binding, selectivity, or
efficacy. It can only be a methods paper with a retrospective sanity check.
That is a legitimate submission at a workshop or a methods track. It is a desk
reject at a venue expecting prospective validation. **Choose the venue
accordingly before writing further.**

### 1.2 The headline numbers contradict each other

| Claim | Value A | Value B | Source | Status |
|---|---|---|---|---|
| MLi-2 redock RMSD | sub-2 Å (founder brief) | 2.9 Å → 0.4 Å after rescoring (deck v3, slide 5) | `LRRK2_Case_Study` | **conflict** |
| Docking score | −9.5 kcal/mol | — | founder brief, "unverified by file" | **unverified** |
| Campaign runtime | 19 minutes | 7.46 h / nine-target docket / one GPU node (deck v3, slide 3) | `LRRK2_Case_Study`, `Capabilities_and_Modules` | **conflict** |
| Model inventory | "12+ validated" | 18 trained (unverified); 24+ modules (deck v3) | `Platform_Overview` | **conflict** |
| Synthesizability check | checked against easy/hard marketed drugs | dataset, n, result: `[TO FILL]` | `Lonnie_Technical_Packet` | **absent** |

`Capabilities_and_Modules` already instructs: "Do not quote either figure until
reconciled." That instruction should be followed in the paper.

**These are not disputes about wording.** Each is a claim about a computation
that either happened, as described, or did not. Resolving them by picking the
more impressive number is fabrication, and a reviewer who asks for the run log
will find it out.

### 1.3 The architecture claim is currently unbacked

The central claim of the draft — that constraining synthesizability at design
time produces a different, better candidate set than filtering afterward — is
testable with a single ablation and has not been run. Without it the paper has
an architecture and a diagram but no result supporting the reason the
architecture differs from an ordinary filter chain.

This is the cheapest gap to close and the one that most changes the paper's
fate. It needs no wet lab.

### 1.4 A single-case redock is not a benchmark

One target, one reference ligand, chosen because a crystal structure exists.
The paper says so plainly; a reviewer will still read it as thin. The
`Section 5` protocol in the draft specifies what would fix it (n ≥ 100 complexes,
top-1/top-5 at 2 Å, PoseBusters filtering, decoy discrimination).

---

## 2. Material that must never enter the submission

- `biothingy` (the BD memo): named government officials, agency targeting
  strategy, IP-rights negotiating posture, pricing and fundraising content.
  Not research content. Excluding it is not optional — including any of it
  would be a conflict-of-interest and confidentiality disaster.
- Company name, founder names, "Lonnie", partner and counterparty names, target
  identities under CDA. The draft is written anonymously and must stay that way
  for double-blind review.
- Internal status vocabulary (`[TO FILL]`, "founder brief", "deck v3") — every
  such reference is replaced by an artifact reference (log file, checkpoint id)
  or the claim is deleted.
- The 929 KB `C_Experiments and Papers_1.pdf` in this folder is a **lecture
  deck on peer review**, not experimental material. It was checked; do not cite
  it.

---

## 3. Work required before submission

Ordered by cost, cheapest first.

1. **Reconcile the register in §1.2.** For each row: locate the log, record the
   value, delete the other. If no log exists, delete the claim. Budget: a few
   hours. Blocks everything else.
2. **Run the constraint ablation (§1.3).** Same target, same budget, generation
   with and without the synthesizability constraint; report route-feasibility
   rate and downstream liability-pass rate for both sets. This is the paper's
   actual contribution and it is a compute job, not a lab job.
3. **Freeze a model manifest.** Name and version every model used, with
   checkpoints. Replaces the 12+/18/24 confusion with a table that cannot be
   misquoted.
4. **Run the route-feasibility reference set.** Name the marketed drugs, the
   easy/hard labels, n, and the adjudication rubric. Report agreement.
5. **Expand pose recovery** to a real benchmark if targeting a main track
   (§1.4). If not, keep the paper as a workshop methods paper and say so in the
   abstract.
6. **Fill every `\pending{}` in `main.tex`** and re-read the abstract: it must
   state, in the abstract itself, that no wet-lab validation exists.

---

## 4. Venue and formatting checklist

- [ ] Pick the venue class file and swap `\documentclass`. The draft uses
      `article` so it compiles anywhere; venues will require their own `.cls`.
- [ ] Page limit: draft is roughly 6 pages two-column with the appendix; check
      the target's limit and whether the appendix counts.
- [ ] Double-blind: strip author block, acknowledgements, funding, and any
      self-citation phrased as "our previous work".
- [ ] Anonymize the artifact links (no company GitHub org, no named S3 bucket).
- [ ] Ethics statement if the venue requires one (no human/animal subjects
      here, so it is a short declaration, but it must be present).
- [ ] Data/code availability: state precisely what is released. "Available on
      request" is increasingly rejected.
- [ ] Compute disclosure: hardware, GPU-hours, and whether any commercial or
      rented compute was used.
- [ ] COI declarations, including any commercial interest in the results —
      this matters more than usual for a paper originating in a startup.
- [ ] Verify every `refs.bib` entry marked `% VERIFY` against the publisher
      record; do not cite from memory at submission time.
- [ ] Re-read for the word "validated". It appears in the source docs referring
      to internal model counts; in a paper it reads as experimental validation.

---

## 5. How to compile

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload `paper/` to Overleaf. Required packages are all standard TeX Live
(`mathptmx`, `booktabs`, `tikz`, `natbib`, `microtype`, `hyperref`).
