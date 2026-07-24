"""
narrative.py  —  Part C: the safe AI narrative
==============================================

generate_narrative(patient_summaries, age_band) takes one patient's last ~30
days of PatientDailySummary and returns a SHORT clinical narrative for a doctor.

Safety-by-design (all four required properties):

  1. NO RAW PHI IN THE PROMPT
     We never send patientId, names, or dates of birth. The patient is referred
     to only by age_band (e.g. "adult_55_64"). Reading DATES stay (a reading
     date is not PHI and the doctor needs it for traceability), but identity is
     scrubbed. `_scrub()` builds a compact, de-identified view.

  2. STRUCTURED, SCHEMA-VALIDATED OUTPUT
     The model must return JSON only. We parse it and validate with the
     `NarrativeOutput` pydantic model. Free text that doesn't validate is
     rejected, not shown.

  3. THE MODEL MAY ABSTAIN
     If coverage is too thin, it must return dataConfidence "insufficient" and
     no invented findings. We also compute coverage ourselves and cap the
     model's confidence so it can't over-claim on sparse data.

  4. GROUNDING — no invented findings
     Every watchItem carries tracedTo {metric, value, date}. After the model
     responds we VERIFY each tracedTo against the actual summaries; any watch
     item whose number doesn't match a real data point is dropped. A concern the
     data can't support does not reach the doctor.

LLM access uses OpenRouter. Set:
    OPENROUTER_API_KEY   and optionally   NARRATIVE_MODEL   (default below).
If no key is present, we fall back to a DETERMINISTIC local builder so the whole
pipeline still runs end-to-end from a clean clone (reproducibility is graded).
The fallback is clearly tagged in the output ("engine": "deterministic_fallback").
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

# Free models to try in order. Verified live July 2026 at openrouter.ai/collections/free-models
# "openrouter/free" is OpenRouter's own auto-router — it picks whichever free model
# is available right now, so it's always the safest first try.
# Override everything with NARRATIVE_MODEL env var (e.g. NARRATIVE_MODEL=anthropic/claude-sonnet-4-6).
FREE_MODEL_FALLBACKS = [
    "google/gemini-2.5-flash-lite",             # confirmed working — try first
    "google/gemma-4-31b-it:free",               # free fallback
    "openrouter/auto",                          # last resort auto-router
]
DEFAULT_MODEL = os.environ.get("NARRATIVE_MODEL")  # None = use FREE_MODEL_FALLBACKS


# --------------------------------------------------------------------------- #
# Output contract (schema-validated)
# --------------------------------------------------------------------------- #
class TracedTo(BaseModel):
    metric: str
    value: Optional[float] = None   # Gemini sometimes returns null; filtered before output
    date: str


class WatchItem(BaseModel):
    rank: int
    concern: str
    tracedTo: TracedTo


class NarrativeOutput(BaseModel):
    summary: str
    watchItems: list[WatchItem] = Field(default_factory=list)
    dataConfidence: Literal["high", "medium", "low", "insufficient"]
    generatedAt: str
    engine: str = "llm"                    # or "deterministic_fallback"
    model_used: Optional[str] = None        # actual model resolved by OpenRouter


# --------------------------------------------------------------------------- #
# PHI scrub + compact view
# --------------------------------------------------------------------------- #
def _scrub(summaries: list[dict], age_band: str) -> dict:
    """De-identified, compact per-day view. No patientId, no names."""
    days = []
    for s in summaries:
        days.append({
            "date": s["date"],
            "steps": s["gait"]["steps"],
            "cadence": s["gait"]["cadence"],
            "symmetry": s["gait"]["symmetry"],
            "rhr": s["cardio"]["rhr"],
            "hrv": s["cardio"]["hrv"],
            "spo2": s["cardio"]["spo2"],
            "sleep_score": s["sleep"]["score"],
            "sleep_hours": s["sleep"]["hours"],
            "recovery": s["load"]["recovery"],
            "strain": s["load"]["strain"],
            "flags": [o["metric"] for o in s["flags"]["outliers"]] +
                     [f"missing:{m}" for m in s["flags"]["missingSources"]],
        })
    return {"ageBand": age_band, "days": days}


# --------------------------------------------------------------------------- #
# Coverage — how much real signal do we actually have?
# --------------------------------------------------------------------------- #
def _coverage(view: dict) -> tuple[float, str]:
    days = view["days"]
    if not days:
        return 0.0, "insufficient"
    keys = ["rhr", "hrv", "spo2", "steps", "sleep_score"]
    filled = sum(1 for d in days for k in keys if d[k] is not None)
    frac = filled / (len(days) * len(keys))
    if frac < 0.25:
        cap = "insufficient"
    elif frac < 0.5:
        cap = "low"
    elif frac < 0.75:
        cap = "medium"
    else:
        cap = "high"
    return round(frac, 3), cap


_CONF_ORDER = {"insufficient": 0, "low": 1, "medium": 2, "high": 3}


def _cap_confidence(model_conf: str, cap: str) -> str:
    return model_conf if _CONF_ORDER[model_conf] <= _CONF_ORDER[cap] else cap


# --------------------------------------------------------------------------- #
# Grounding — drop any watch item whose number isn't in the data
# --------------------------------------------------------------------------- #
def _index(view: dict) -> dict[tuple[str, str], float]:
    idx = {}
    for d in view["days"]:
        for m in ["rhr", "hrv", "spo2", "steps", "cadence", "symmetry",
                  "sleep_score", "sleep_hours", "recovery", "strain"]:
            if d[m] is not None:
                idx[(m, d["date"])] = float(d[m])
    return idx


def _verify_grounding(items: list[WatchItem], view: dict, tol: float = 2.0
                      ) -> list[WatchItem]:
    idx = _index(view)
    kept = []
    for it in items:
        key = (it.tracedTo.metric, it.tracedTo.date)
        real = idx.get(key)
        if real is not None and abs(real - it.tracedTo.value) <= max(tol, abs(real) * 0.02):
            kept.append(it)
        # else: silently dropped — an ungrounded concern never reaches the doctor
    for i, it in enumerate(sorted(kept, key=lambda x: x.rank), start=1):
        it.rank = i
    return kept


# --------------------------------------------------------------------------- #
# LLM call (OpenRouter)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a careful clinical-support assistant summarizing \
de-identified wearable and gait data for a physician. You must:
- Write a 3-5 sentence plain-language summary.
- List 0-3 ranked "things to watch", each traceable to ONE real data point.
- Use ONLY numbers present in the provided data. Never invent a value, metric, \
or date. If a value is null/missing, do not report it.
- If coverage is too thin to be trustworthy, set dataConfidence to \
"insufficient" and return an empty watchItems list rather than guessing.
Return ONLY valid JSON, no markdown, matching exactly:
{"summary": str,
 "watchItems": [{"rank": int, "concern": str,
                 "tracedTo": {"metric": str, "value": number, "date": "YYYY-MM-DD"}}],
 "dataConfidence": "high"|"medium"|"low"|"insufficient"}"""


def _call_openrouter(view: dict) -> Optional[dict]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    # Build the list of models to try: explicit env override first, then free list.
    models_to_try = [DEFAULT_MODEL] if DEFAULT_MODEL else FREE_MODEL_FALLBACKS

    for model in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=800,      # narrative is short JSON; 800 is plenty
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(view)},
                ],
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown fences
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            # Extract first JSON object if model added preamble
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("no JSON object found in response")
            text = text[start:end]
            result = json.loads(text)
            # resp.model is the actual model OpenRouter resolved to
            # (e.g. "openrouter/auto" resolves to the real model name)
            resolved = getattr(resp, "model", model)
            result["model_used"] = resolved
            print(f"    [narrative] used model: {resolved}")
            return result
        except Exception as e:
            # 404 = model gone from free tier; try next. Other errors also try next.
            print(f"    [narrative] {model} failed ({type(e).__name__}); trying next")

    print("    [narrative] all models failed; using deterministic fallback")
    return None


# --------------------------------------------------------------------------- #
# Deterministic fallback — grounded, abstains, schema-valid, no network needed
# --------------------------------------------------------------------------- #
def _fallback(view: dict, cap: str) -> dict:
    days = [d for d in view["days"]]
    watch = []

    def latest(metric):
        for d in reversed(days):
            if d[metric] is not None:
                return d[metric], d["date"]
        return None, None

    # HRV downward trend over the last week (a real, checkable signal)
    hrv = [(d["date"], d["hrv"]) for d in days if d["hrv"] is not None][-7:]
    if len(hrv) >= 4 and hrv[-1][1] < hrv[0][1]:
        watch.append(WatchItem(rank=len(watch) + 1,
                     concern="HRV lower at end of the last week than at its start.",
                     tracedTo=TracedTo(metric="hrv", value=float(hrv[-1][1]),
                                       date=hrv[-1][0])))
    # Any low SpO2 reading
    lo = [(d["date"], d["spo2"]) for d in days if d["spo2"] is not None and d["spo2"] < 94]
    if lo:
        watch.append(WatchItem(rank=len(watch) + 1,
                     concern="At least one day with SpO2 below 94%.",
                     tracedTo=TracedTo(metric="spo2", value=float(lo[-1][1]),
                                       date=lo[-1][0])))
    # Elevated resting HR
    rhr_vals = [d["rhr"] for d in days if d["rhr"] is not None]
    if rhr_vals:
        rv, rd = latest("rhr")
        base = sum(rhr_vals) / len(rhr_vals)
        if rv and rv > base + 8:
            watch.append(WatchItem(rank=len(watch) + 1,
                         concern="Most recent resting HR is above the 30-day average.",
                         tracedTo=TracedTo(metric="rhr", value=float(rv), date=rd)))

    n_gait = sum(1 for d in days if d["steps"] is not None)
    conf = "insufficient" if cap == "insufficient" else ("low" if not watch else cap)
    summary = (
        f"De-identified {view['ageBand']} patient with {len(days)} days of data; "
        f"{n_gait} days have insole gait. "
        + ("Cardio and sleep coverage is adequate for a high-level view. "
           if cap in ("medium", "high") else
           "Coverage is limited, so findings are tentative. ")
        + ("No strong concerns stand out from the available numbers."
           if not watch else
           f"{len(watch)} item(s) flagged below, each tied to a specific reading.")
    )
    return NarrativeOutput(
        summary=summary, watchItems=watch[:3], dataConfidence=conf,
        generatedAt=datetime.now(timezone.utc).isoformat(),
        engine="deterministic_fallback",
    ).model_dump()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_narrative(patient_summaries: list[dict], age_band: str = "unknown") -> dict:
    view = _scrub(patient_summaries, age_band)
    frac, cap = _coverage(view)

    raw = _call_openrouter(view)
    if raw is None:
        os.environ["NARRATIVE_MODE_USED"] = "deterministic_fallback"
        return _fallback(view, cap)

    # Validate the model's JSON against the schema; fall back if it doesn't fit.
    try:
        raw.setdefault("generatedAt", datetime.now(timezone.utc).isoformat())
        # Strip fields the model may have added that aren't in our schema
        allowed = {"summary", "watchItems", "dataConfidence", "generatedAt"}
        raw_clean = {k: v for k, v in raw.items() if k in allowed}
        parsed = NarrativeOutput(**raw_clean)
    except ValidationError as e:
        print(f"    [narrative] schema validation failed: {e}; using deterministic fallback")
        os.environ["NARRATIVE_MODE_USED"] = "deterministic_fallback"
        return _fallback(view, cap)

    # Drop items where the model left value=null (can't ground a null)
    parsed.watchItems = [it for it in parsed.watchItems if it.tracedTo.value is not None]
    parsed.watchItems = _verify_grounding(parsed.watchItems, view)  # grounding gate
    parsed.dataConfidence = _cap_confidence(parsed.dataConfidence, cap)  # anti-overclaim
    if cap == "insufficient":
        parsed.watchItems = []
    parsed.engine = "llm"
    parsed.model_used = raw.get("model_used")   # set by _call_openrouter
    os.environ["NARRATIVE_MODE_USED"] = "llm"
    return parsed.model_dump()