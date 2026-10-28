"""LLM extraction of FDA warning letters, and the audit that bounds its error.

Section II-B claims a fixed JSON schema, an FEI link, and a hand audit with
reported field-level accuracy and entity-link precision. This module
implements the first two and the sampling/scoring half of the third, so the
numbers in the paper come from `audit_score`, not from a guess.

The extractor speaks the OpenAI-compatible chat API. Set:

    OPENAI_API_KEY   required for live extraction
    OPENAI_BASE_URL  optional, defaults to https://api.freebuff.com/api/v1/paid
    OPENAI_MODEL     optional, defaults to the cheapest capable model

No key is read from, or written to, any file in this repository. Extraction is
cached per letter so re-runs are free and reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SCHEMA = {
    "type": "object",
    "properties": {
        "firm_name": {"type": "string"},
        "firm_address": {"type": "string"},
        "letter_date": {"type": "string", "description": "YYYY-MM-DD"},
        "fei_numbers": {"type": "array", "items": {"type": "string"}},
        "cgmp_categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "sterility_assurance",
                    "contamination_control",
                    "data_integrity",
                    "laboratory_controls",
                    "process_validation",
                    "quality_systems",
                    "other",
                ],
            },
        },
        "import_alert_referenced": {"type": "boolean"},
    },
    "required": [
        "firm_name", "firm_address", "letter_date", "fei_numbers",
        "cgmp_categories", "import_alert_referenced",
    ],
}

PROMPT = """You extract structured fields from a U.S. FDA warning letter.

Return ONLY a JSON object matching this schema (no prose, no markdown fences):
{schema}

Rules:
- letter_date is the date printed on the letter, formatted YYYY-MM-DD.
- fei_numbers contains FDA Establishment Identifier numbers exactly as printed.
  Return an empty array if none appear.
- cgmp_categories uses only the enum values listed, judged from the violations
  actually cited in the letter.
- import_alert_referenced is true only if the letter mentions an import alert.

Letter text:
\"\"\"
{text}
\"\"\"
"""


@dataclass
class Extracted:
    firm_name: str
    firm_address: str
    letter_date: str
    fei_numbers: list[str]
    cgmp_categories: list[str]
    import_alert_referenced: bool


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _call_llm(prompt: str, *, timeout: int = 120) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Live extraction needs it; the audit and "
            "graph code do not."
        )
    base = os.environ.get("OPENAI_BASE_URL", "https://api.freebuff.com/api/v1/paid")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def extract_letter(text: str, cache_dir: Path) -> Extracted:
    """Extract one letter, caching the raw model output on disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(text)}.json"
    if cache_file.exists():
        return Extracted(**json.loads(cache_file.read_text(encoding="utf-8")))
    raw = _call_llm(PROMPT.format(schema=json.dumps(SCHEMA), text=text[:12000]))
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    missing = [key for key in SCHEMA["required"] if key not in data]
    if missing:
        raise ValueError(f"model output missing fields {missing}")
    record = Extracted(**{key: data[key] for key in SCHEMA["required"]})
    cache_file.write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")
    return record


def extract_batch(letters_path: Path, out_path: Path, cache_dir: Path) -> int:
    """Extract a JSONL of {text, ...} records into the ingest format."""
    rows = [
        json.loads(line)
        for line in letters_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            try:
                record = extract_letter(row["text"], cache_dir)
            except Exception as exc:  # keep going; the audit counts failures
                fh.write(json.dumps({"error": str(exc), "firm": row.get("firm", "")}) + "\n")
                continue
            fh.write(
                json.dumps(
                    {
                        "letter_date": record.letter_date,
                        "firm": record.firm_name,
                        "fei": record.fei_numbers[0] if record.fei_numbers else None,
                        "cgmp_categories": record.cgmp_categories,
                        "import_alert": record.import_alert_referenced,
                        "text": row.get("text", "")[:2000],
                    }
                )
                + "\n"
            )
            written += 1
    return written


# --- hand audit -------------------------------------------------------------


def audit_sample(extracted_path: Path, out_csv: Path, *, n: int, seed: int = 0) -> int:
    """Draw a random sample for hand-checking into a blank CSV.

    Columns mirror the fields that matter for modelling; the auditor fills
    `gold_*` and `link_correct` (1/0/blank).
    """
    rows = [
        json.loads(line)
        for line in extracted_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "error" not in line
    ]
    rng = random.Random(seed)
    sample = rng.sample(rows, k=min(n, len(rows)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "firm", "letter_date", "fei",
                "pred_cgmp_categories", "pred_import_alert",
                "gold_firm", "gold_letter_date", "gold_fei",
                "gold_cgmp_categories", "gold_import_alert",
                "link_correct",
            ]
        )
        for row in sample:
            writer.writerow(
                [
                    row.get("firm", ""),
                    row.get("letter_date", ""),
                    row.get("fei") or "",
                    "|".join(row.get("cgmp_categories") or []),
                    int(bool(row.get("import_alert"))),
                    "", "", "", "", "", "",
                ]
            )
    return len(sample)


def audit_score(audit_csv: Path) -> dict:
    """Field-level accuracy and entity-link precision over a completed audit.

    A row counts toward link precision only when `gold_fei` and `link_correct`
    are filled. Field accuracy is computed over rows where the gold value was
    entered, so a partially completed audit reports honest denominators.
    """
    fields = {
        "firm": 0, "letter_date": 0, "cgmp": 0, "import_alert": 0,
    }
    totals = {key: 0 for key in fields}
    link_ok = link_total = 0

    with audit_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("gold_firm"):
                totals["firm"] += 1
                fields["firm"] += int(row["firm"].strip() == row["gold_firm"].strip())
            if row.get("gold_letter_date"):
                totals["letter_date"] += 1
                fields["letter_date"] += int(row["letter_date"].strip() == row["gold_letter_date"].strip())
            if row.get("gold_cgmp_categories"):
                totals["cgmp"] += 1
                pred = {x for x in row["pred_cgmp_categories"].split("|") if x}
                gold = {x for x in row["gold_cgmp_categories"].split("|") if x}
                fields["cgmp"] += int(pred == gold)
            if row.get("gold_import_alert") != "":
                totals["import_alert"] += 1
                fields["import_alert"] += int(
                    (row["pred_import_alert"].strip() == "1")
                    == (row["gold_import_alert"].strip() in {"1", "true", "yes", "True"})
                )
            if row.get("link_correct") not in (None, ""):
                link_total += 1
                link_ok += int(row["link_correct"].strip() in {"1", "true", "yes"})

    return {
        "n_scored": {key: totals[key] for key in totals},
        "field_accuracy": {
            key: (fields[key] / totals[key] if totals[key] else None) for key in fields
        },
        "entity_link_precision": (link_ok / link_total if link_total else None),
        "entity_link_n": link_total,
    }
