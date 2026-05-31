"""
Finalize the strict literature audit:

  1. Read `reports/cascade_pairs/audit_decisions_validated.yaml` (produced
     by `validate_audit.py`).
  2. Read `reports/cascade_pairs/audit_digest.csv` for citation URLs.
  3. Emit two artifacts:
       - `reports/cascade_pairs/cytokine_axes_audited.csv` — refined labels
         in machine-readable form for the retally script.
       - `reports/cascade_pairs/audit_log.md` — user-readable per-pair
         markdown grouped by pair_status with verbatim quotes.

Exit 2 if validated YAML missing.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parent.parent
VALIDATED_YAML = REPO / "reports/cascade_pairs/audit_decisions_validated.yaml"
DIGEST_CSV = REPO / "reports/cascade_pairs/audit_digest.csv"
OUT_CSV = REPO / "reports/cascade_pairs/cytokine_axes_audited.csv"
OUT_MD = REPO / "reports/cascade_pairs/audit_log.md"


STATUS_ORDER = [
    "DIRECTIONAL_a_to_b",
    "DIRECTIONAL_b_to_a",
    "BIDIRECTIONAL",
    "DIRECTIONAL_a_to_b_NOISY",
    "DIRECTIONAL_b_to_a_NOISY",
    "PARTIAL_INHIBITORY",
    "WEAK_a_to_b",
    "WEAK_b_to_a",
    "LOW_CONFIDENCE",
    "UNKNOWN",
]


def load_yaml(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["audits"]


def load_digest(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["axis_a"], row["axis_b"])] = row
    return out


def write_csv(audits: list[dict], digest: dict, path: Path) -> None:
    fieldnames = [
        "axis_a", "axis_b",
        "a_to_b_label", "a_to_b_rule_id", "a_to_b_quote",
        "a_to_b_n_primary", "a_to_b_n_secondary",
        "b_to_a_label", "b_to_a_rule_id", "b_to_a_quote",
        "b_to_a_n_primary", "b_to_a_n_secondary",
        "pair_status", "expected_sign", "audit_direction",
        "original_direction", "tag_changed",
        "counts_in_benchmark",
    ]
    rows = []
    for a in audits:
        ax_a, ax_b = a["axis"]
        d = a["derived"]
        a_b = a["a_to_b"]
        b_a = a["b_to_a"]
        rows.append({
            "axis_a": ax_a, "axis_b": ax_b,
            "a_to_b_label": a_b["label"], "a_to_b_rule_id": a_b["rule_id"],
            "a_to_b_quote": a_b["quote"],
            "a_to_b_n_primary": a_b["n_citations_primary"],
            "a_to_b_n_secondary": a_b["n_citations_secondary"],
            "b_to_a_label": b_a["label"], "b_to_a_rule_id": b_a["rule_id"],
            "b_to_a_quote": b_a["quote"],
            "b_to_a_n_primary": b_a["n_citations_primary"],
            "b_to_a_n_secondary": b_a["n_citations_secondary"],
            "pair_status": d["pair_status"],
            "expected_sign": "" if d["expected_sign"] is None else d["expected_sign"],
            "audit_direction": d["audit_direction"] or "",
            "original_direction": d["original_direction"],
            "tag_changed": d["tag_changed"],
            "counts_in_benchmark": d["counts_in_benchmark"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)", flush=True)


def write_md(audits: list[dict], digest: dict, path: Path) -> None:
    by_status: dict[str, list[dict]] = defaultdict(list)
    for a in audits:
        by_status[a["derived"]["pair_status"]].append(a)

    lines = []
    lines.append("# Strict literature audit log — 53 evaluable Oesinghaus axes")
    lines.append("")
    lines.append(
        "Per-direction labels: `POSITIVE_STRONG` (R1), `POSITIVE_WEAK` (R2), "
        "`INHIBITORY` (R3), `UNKNOWN` (R4), `POSITIVE_STRONG` (R5 = pre-registered)."
    )
    lines.append("Per-pair status derived from the two directional labels; see plan §Per-pair status.")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    in_bench = sum(1 for a in audits if a["derived"]["counts_in_benchmark"])
    n_changed = sum(1 for a in audits if a["derived"]["tag_changed"])
    lines.append(f"- Total axes audited: **{len(audits)}**")
    lines.append(f"- Axes counted in benchmark accuracy: **{in_bench}**")
    lines.append(f"- Of those, tag flipped vs original `cytokine_axes.csv`: **{n_changed}**")
    lines.append("")
    lines.append("## Per-pair-status counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for st in STATUS_ORDER:
        if st in by_status:
            lines.append(f"| {st} | {len(by_status[st])} |")
    lines.append("")

    for st in STATUS_ORDER:
        if st not in by_status:
            continue
        lines.append(f"## {st}")
        lines.append("")
        # Build a compact table per category
        lines.append(
            "| axis | original_tag | tag_changed | a_to_b | b_to_a | "
            "a_to_b quote | b_to_a quote | primary cits |"
        )
        lines.append("|---|---|:-:|---|---|---|---|---|")
        for a in sorted(by_status[st], key=lambda x: (x["axis"][0], x["axis"][1])):
            ax_a, ax_b = a["axis"]
            d = a["derived"]
            a_b = a["a_to_b"]
            b_a = a["b_to_a"]
            tot_primary = int(a_b["n_citations_primary"]) + int(b_a["n_citations_primary"])
            tc = "✓" if d["tag_changed"] else ""
            # truncate quotes for the table
            def _trunc(s: str, n: int = 80) -> str:
                s = (s or "").replace("|", "\\|").replace("\n", " ")
                return s if len(s) <= n else s[:n] + "..."
            lines.append(
                f"| {ax_a} / {ax_b} | {d['original_direction']} | {tc} | "
                f"{a_b['label']} ({a_b['rule_id']}) | "
                f"{b_a['label']} ({b_a['rule_id']}) | "
                f"{_trunc(a_b['quote'])} | {_trunc(b_a['quote'])} | "
                f"{tot_primary} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path}", flush=True)


def main():
    if not VALIDATED_YAML.exists():
        print(f"FATAL: run validate_audit.py first; {VALIDATED_YAML} missing",
              file=sys.stderr)
        sys.exit(2)

    audits = load_yaml(VALIDATED_YAML)
    digest = load_digest(DIGEST_CSV)
    write_csv(audits, digest, OUT_CSV)
    write_md(audits, digest, OUT_MD)


if __name__ == "__main__":
    main()
