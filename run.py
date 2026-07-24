"""
run.py  —  end-to-end entry point
=================================

    python run.py                # generate -> fuse -> narrate -> write outputs
    python run.py --check-idempotency   # run the pipeline twice, prove identical
    python run.py --no-narrative        # skip the LLM step (pipeline + outputs only)

Outputs
    outputs/summaries/<patientId>.json   list[PatientDailySummary]  (one per day)
    outputs/narratives/<patientId>.json  the safe AI narrative for that patient

Idempotency
    `computedAt` is PINNED to a constant (COMPUTED_AT) so the whole run is a pure
    function of the raw inputs and the JSON is byte-identical across runs. In
    production this would be the real wall-clock UTC; we pin it here purely so
    reproducibility is graded on the data, not the clock. `--check-idempotency`
    hashes the summary files, reruns, and compares.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import generate_data
from pipeline import run_pipeline

COMPUTED_AT = "2025-06-30T00:00:00Z"      # pinned for reproducible, hashable output
OUT_SUMMARIES = os.path.join("outputs", "summaries")
OUT_NARRATIVES = os.path.join("outputs", "narratives")


def _dump(summaries) -> str:
    """Deterministic JSON (sorted keys) so repeated runs are byte-identical."""
    rows = [s.model_dump() for s in summaries]
    return json.dumps(rows, sort_keys=True, indent=2, ensure_ascii=False)


def write_summaries(result) -> None:
    os.makedirs(OUT_SUMMARIES, exist_ok=True)
    for pid, summaries in result.items():
        with open(os.path.join(OUT_SUMMARIES, f"{pid}.json"), "w") as f:
            f.write(_dump(summaries))
    print(f"  wrote {len(result)} summary files -> {OUT_SUMMARIES}/")


def write_narratives(result) -> None:
    from narrative import generate_narrative
    import pandas as pd
    meta = pd.read_csv(os.path.join(generate_data.RAW_DIR, "patients.csv"))
    band = dict(zip(meta["patient_id"], meta["age_band"]))
    os.makedirs(OUT_NARRATIVES, exist_ok=True)
    modes, models = [], set()
    for pid, summaries in result.items():
        narr = generate_narrative([s.model_dump() for s in summaries],
                                  age_band=band.get(pid, "unknown"))
        engine = narr.get("engine", "unknown")
        modes.append(engine)
        model_used = narr.get("model_used")
        if model_used and engine == "llm":
            models.add(model_used)
        with open(os.path.join(OUT_NARRATIVES, f"{pid}.json"), "w") as f:
            f.write(json.dumps(narr, sort_keys=True, indent=2, ensure_ascii=False))
    llm_count = sum(1 for m in modes if m == "llm")
    mode_summary = f"llm ({llm_count}/{len(modes)})" if llm_count else "deterministic_fallback"
    model_str = ", ".join(sorted(models)) if models else "n/a"
    print(f"  wrote {len(result)} narrative files -> {OUT_NARRATIVES}/")
    print(f"  mode: {mode_summary}  |  model(s): {model_str}")


def check_idempotency() -> None:
    generate_data.main()
    r1 = run_pipeline(COMPUTED_AT)
    h1 = {p: hashlib.md5(_dump(s).encode()).hexdigest() for p, s in r1.items()}
    generate_data.main()
    r2 = run_pipeline(COMPUTED_AT)
    h2 = {p: hashlib.md5(_dump(s).encode()).hexdigest() for p, s in r2.items()}
    assert h1 == h2, "NON-IDEMPOTENT: output changed across runs!"
    print("  idempotency OK — summary hashes identical across two full runs:")
    for p in sorted(h1):
        print(f"    {p}  {h1[p]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-narrative", action="store_true")
    ap.add_argument("--check-idempotency", action="store_true")
    args = ap.parse_args()

    if args.check_idempotency:
        print("[idempotency check]")
        check_idempotency()
        return

    print("[1/3] generating synthetic raw data")
    generate_data.main()

    print("\n[2/3] running fusion + cleaning pipeline")
    result = run_pipeline(COMPUTED_AT)
    write_summaries(result)

    if not args.no_narrative:
        print("\n[3/3] generating safe AI narratives")
        write_narratives(result)
    else:
        print("\n[3/3] skipped (--no-narrative)")

    print("\nDone.")


if __name__ == "__main__":
    main()