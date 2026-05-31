"""
Validate `reports/cascade_pairs/audit_decisions.yaml`:

  * Every of the 53 evaluable axes has exactly one decision block.
  * Both `a_to_b` and `b_to_a` directions populated per axis.
  * Label / rule_id pairing is consistent:
      POSITIVE_STRONG  → R1 or R5 (pre-registered)
      POSITIVE_WEAK    → R2
      INHIBITORY       → R3
      UNKNOWN          → R4
  * Every quote is a verbatim substring of either source summary
    (a_to_b_summary or b_to_a_summary from audit_digest.csv), with two
    documented exceptions:
      - "(direction not mentioned in summary)"
      - "Pre-registered KNOWN_CASCADE"
  * POSITIVE_STRONG with rule_id=R1 requires n_citations_primary >= 1.
  * POSITIVE_STRONG with rule_id=R5 bypasses that (pre-registered).

If all checks pass, derive `pair_status`, `expected_sign`,
`counts_in_benchmark`, `tag_changed` for each axis and emit the
validated YAML to `reports/cascade_pairs/audit_decisions_validated.yaml`.

Exit code 0 on success, 2 on any failure.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parent.parent
DECISIONS_PATH = REPO / "reports/cascade_pairs/audit_decisions.yaml"
DIGEST_PATH = REPO / "reports/cascade_pairs/audit_digest.csv"
OUT_PATH = REPO / "reports/cascade_pairs/audit_decisions_validated.yaml"


NOT_MENTIONED = "(direction not mentioned in summary)"
PRE_REG_QUOTE = "Pre-registered KNOWN_CASCADE"

VALID_LABELS = {"POSITIVE_STRONG", "POSITIVE_WEAK", "INHIBITORY", "UNKNOWN"}
LABEL_TO_RULES = {
    "POSITIVE_STRONG": {"R1", "R5"},
    "POSITIVE_WEAK": {"R2"},
    "INHIBITORY": {"R3"},
    "UNKNOWN": {"R4"},
}


# ----------------------------------------------------------------------- #
# Loaders
# ----------------------------------------------------------------------- #

def load_decisions(path: Path) -> list[dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data["audits"]


def load_digest(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["axis_a"], row["axis_b"])] = row
    return out


# ----------------------------------------------------------------------- #
# Validation
# ----------------------------------------------------------------------- #

def validate_one(
    audit: dict,
    digest_row: dict,
    errors: list[str],
) -> None:
    a, b = audit["axis"]
    axis_label = f"{a} / {b}"

    if "a_to_b" not in audit or "b_to_a" not in audit:
        errors.append(f"{axis_label}: missing a_to_b or b_to_a block")
        return

    a_sum = digest_row.get("a_to_b_summary", "") or ""
    b_sum = digest_row.get("b_to_a_summary", "") or ""
    orig_sum = digest_row.get("original_summary", "") or ""

    for dir_name in ("a_to_b", "b_to_a"):
        block = audit[dir_name]
        for field in ("label", "rule_id", "quote",
                      "n_citations_primary", "n_citations_secondary"):
            if field not in block:
                errors.append(f"{axis_label} [{dir_name}]: missing field '{field}'")
                return

        lbl = block["label"]
        rid = block["rule_id"]
        if lbl not in VALID_LABELS:
            errors.append(f"{axis_label} [{dir_name}]: bad label '{lbl}'")
        if lbl in LABEL_TO_RULES and rid not in LABEL_TO_RULES[lbl]:
            errors.append(
                f"{axis_label} [{dir_name}]: label={lbl} but rule_id={rid} "
                f"(expected one of {sorted(LABEL_TO_RULES[lbl])})"
            )

        quote = block["quote"]
        if quote in (NOT_MENTIONED, PRE_REG_QUOTE):
            pass
        else:
            if (quote not in a_sum) and (quote not in b_sum) and (quote not in orig_sum):
                errors.append(
                    f"{axis_label} [{dir_name}]: quote not a verbatim substring "
                    f"of either summary:\n      quote: {quote!r}"
                )

        if lbl == "POSITIVE_STRONG" and rid == "R1":
            if int(block["n_citations_primary"]) < 1:
                errors.append(
                    f"{axis_label} [{dir_name}]: POSITIVE_STRONG (R1) requires "
                    f"n_citations_primary >= 1; got {block['n_citations_primary']}"
                )


# ----------------------------------------------------------------------- #
# Derive per-pair status
# ----------------------------------------------------------------------- #

def derive_pair_status(audit: dict) -> dict:
    """Compute pair_status, expected_sign, counts_in_benchmark from a/b labels."""
    a_lab = audit["a_to_b"]["label"]
    b_lab = audit["b_to_a"]["label"]

    # INHIBITORY in either direction → PARTIAL_INHIBITORY
    if "INHIBITORY" in (a_lab, b_lab):
        return dict(pair_status="PARTIAL_INHIBITORY", expected_sign=None, counts_in_benchmark=False)

    if a_lab == "POSITIVE_STRONG" and b_lab == "POSITIVE_STRONG":
        return dict(pair_status="BIDIRECTIONAL", expected_sign=None, counts_in_benchmark=False)
    if a_lab == "POSITIVE_STRONG" and b_lab == "UNKNOWN":
        return dict(pair_status="DIRECTIONAL_a_to_b", expected_sign=+1, counts_in_benchmark=True)
    if a_lab == "UNKNOWN" and b_lab == "POSITIVE_STRONG":
        return dict(pair_status="DIRECTIONAL_b_to_a", expected_sign=-1, counts_in_benchmark=True)
    if a_lab == "POSITIVE_STRONG" and b_lab == "POSITIVE_WEAK":
        return dict(pair_status="DIRECTIONAL_a_to_b_NOISY", expected_sign=+1, counts_in_benchmark=False)
    if a_lab == "POSITIVE_WEAK" and b_lab == "POSITIVE_STRONG":
        return dict(pair_status="DIRECTIONAL_b_to_a_NOISY", expected_sign=-1, counts_in_benchmark=False)
    if a_lab == "POSITIVE_WEAK" and b_lab == "UNKNOWN":
        return dict(pair_status="WEAK_a_to_b", expected_sign=+1, counts_in_benchmark=False)
    if a_lab == "UNKNOWN" and b_lab == "POSITIVE_WEAK":
        return dict(pair_status="WEAK_b_to_a", expected_sign=-1, counts_in_benchmark=False)
    if a_lab == "POSITIVE_WEAK" and b_lab == "POSITIVE_WEAK":
        return dict(pair_status="LOW_CONFIDENCE", expected_sign=None, counts_in_benchmark=False)
    if a_lab == "UNKNOWN" and b_lab == "UNKNOWN":
        return dict(pair_status="UNKNOWN", expected_sign=None, counts_in_benchmark=False)

    # Fallthrough
    return dict(pair_status=f"OTHER_{a_lab}_{b_lab}", expected_sign=None, counts_in_benchmark=False)


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    if not DECISIONS_PATH.exists():
        print(f"FATAL: {DECISIONS_PATH} missing", file=sys.stderr)
        sys.exit(2)
    if not DIGEST_PATH.exists():
        print(f"FATAL: {DIGEST_PATH} missing", file=sys.stderr)
        sys.exit(2)

    audits = load_decisions(DECISIONS_PATH)
    digest = load_digest(DIGEST_PATH)

    # Axis coverage check
    audit_axes = set((tuple(a["axis"])) for a in audits)
    digest_axes = set(digest.keys())
    missing = digest_axes - audit_axes
    extra = audit_axes - digest_axes
    fatal = []
    if missing:
        fatal.append(f"Digest axes without decision: {sorted(missing)}")
    if extra:
        fatal.append(f"Decisions not in digest: {sorted(extra)}")

    errors: list[str] = []
    for audit in audits:
        a, b = audit["axis"]
        if (a, b) not in digest:
            continue  # already flagged above
        validate_one(audit, digest[(a, b)], errors)

    if fatal or errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for line in fatal:
            print(f"  {line}", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(2)

    # Derive per-pair status and write validated YAML
    for audit in audits:
        derived = derive_pair_status(audit)
        # tag_changed: did our DIRECTIONAL_* match the original tag?
        original = digest[tuple(audit["axis"])].get("original_direction", "")
        pair_status = derived["pair_status"]
        if pair_status == "DIRECTIONAL_a_to_b":
            audit_dir = "a_to_b"
        elif pair_status == "DIRECTIONAL_b_to_a":
            audit_dir = "b_to_a"
        else:
            audit_dir = None
        if audit_dir is not None and original in ("a_to_b", "b_to_a"):
            tag_changed = (audit_dir != original)
        else:
            tag_changed = False
        audit["derived"] = {
            **derived,
            "audit_direction": audit_dir,
            "original_direction": original,
            "tag_changed": tag_changed,
        }

    # Summary
    by_status = {}
    n_tag_changed = 0
    for audit in audits:
        st = audit["derived"]["pair_status"]
        by_status[st] = by_status.get(st, 0) + 1
        if audit["derived"]["tag_changed"]:
            n_tag_changed += 1

    print("Per-pair status counts:")
    for st in sorted(by_status):
        n = by_status[st]
        print(f"  {st:32s} {n:3d}")
    print(f"\nTotal axes:        {len(audits)}")
    in_bench = sum(1 for a in audits if a["derived"]["counts_in_benchmark"])
    print(f"In benchmark:      {in_bench}")
    print(f"Tag changed (in directional benchmark): {n_tag_changed}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        yaml.dump({"audits": audits}, f, sort_keys=False, default_flow_style=False, width=120)
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
