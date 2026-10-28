# Placeholders to fill before submitting

The results are complete. What remains is administrative, not scientific.

```bash
grep -n '\\ph{' embc/main.tex
```

## Author block

| Where | Needs |
|---|---|
| `\IEEEauthorblockA` ×2 | Department, institution, city, state ZIP, country per author |
| same | Email address per author |

EMBC 2027 allows up to 35 authors; list them in the portal exactly as in the paper.

## Acknowledgment

| Placeholder | Needs |
|---|---|
| Funding / compute / data access | one sentence naming sources, or "no external funding" |
| AI disclosure (tool name) | the assistant used for drafting text and code, and confirmation of the sections affected |

The AI disclosure is mandatory: undeclared generative-AI use is grounds for
rejection. Current text already states that manuscript and code were drafted with
an AI assistant and that figures are code output; add the tool name.

## Conclusion

| Placeholder | Needs |
|---|---|
| `\ph{URL}` | public repository for the code |

## Checks before upload

- [ ] Technical content ends on **page 4**; page 5 holds only acknowledgment and references.
- [ ] Reference count ≤ 20 (currently 12, all cited).
- [ ] No margin, font or column changes — stock `IEEEtran[conference,letterpaper]`.
- [ ] Author affiliations filled in.
- [ ] Reciprocal reviewer designated in the portal, or a verified exemption declared.
- [ ] Theme mapped: **Biomedical & Health Informatics**.
- [ ] `python3 -m pytest tests -q` passes (20 tests).
- [ ] Numbers in Tables I–III match a fresh run of `realdata` and `casecontrol`.
- [ ] Section V still reports **no** performance figure — the only measured
      numbers in the paper are the cross-sectional ones and Table I.
- [ ] References spot-checked against publisher records — carried over from the
      original manuscript and not re-verified here.

## Numbers that are deliberately absent

- **Establishment (FEI) counts and facility-level edges.** These need the
  NDC→FEI mapping from DailyMed SPL, which is not built. The paper says so and
  draws no two-hop conclusion. Do not add a facility claim without that mapping.
- **Forecast performance (AUPRC, lead time).** The labels do not exist, so these
  numbers do not either. Section V specifies the model, the baselines and the
  protocol and deliberately reports none of them; filling that gap with a fixture
  or synthetic run would misrepresent the paper. Reintroducing numbers means
  reintroducing the label problem first.
