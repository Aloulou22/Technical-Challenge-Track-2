# eSteps Health — Track 2 · Data Scientist Challenge

Fusing messy multi-source health-sensor data into one clean, trustworthy daily view a clinician can act on.

**Candidate:** Fares Aloulou · ISIM Monastir (AI, 2nd year CSE)
**Stack:** Python 3.12 · pandas · pydantic v2 · Streamlit · OpenRouter (Gemini 2.5 Flash Lite)

---

## Run it

```bash
pip install -r requirements.txt
python run.py                       # generate → fuse → narrate → write outputs
```

Optional:

```bash
export OPENROUTER_API_KEY=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 # live LLM narrative (falls back to deterministic if unset) - For testing purposes only, you can set a shared API key
python run.py --check-idempotency   # run twice, prove byte-identical output
pytest -v                           # 3 unit tests: timezone, dedup, outlier
streamlit run dashboard.py          # bonus: patient dashboard
```

Outputs land in `outputs/summaries/<patientId>.json` and `outputs/narratives/<patientId>.json`.

---

## Pipeline

```
  Insole CSV            Ring CSV              Watch CSV
  1 row / day           96 rows / night       48 rows / day
  gaps + spikes         +02:00 offset bug     fraction SpO2, dupes, drift day
       │                     │                      │
       ▼                     ▼                      ▼
  outlier flags        ring_daily()           watch_daily()
  physio + IQR         tz → local day         dedup · spo2×100
                       wake attribution       drift detection
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             ▼
                     build_patient()
          ┌──────────┬──────────┬─────────┬────────┐
          │   GAIT   │  CARDIO  │  SLEEP  │  LOAD  │
          │ insole   │ring>watch│  ring   │ watch  │
          └──────────┴──────────┴─────────┴────────┘
                             ▼
                   pydantic validation
                  (no silent NaNs allowed)
                             ▼
                  PatientDailySummary  ──►  narrative.py  ──►  NarrativeOutput
                  versioned · source-tagged                    PHI-safe · grounded
```

---

## Source-priority rules and why

| Field | Source | Why |
|---|---|---|
| `gait.*` | **Insole only** | Insole is ground truth for movement. A wearable step count is a proxy, not a clinical reading — never substituted. No insole row → `missing`. |
| `cardio.rhr` · `cardio.hrv` | **Ring → Watch** | The ring is a dedicated overnight sensor worn through sleep; it's more reliable at rest than a general-purpose watch. Watch is fallback only. |
| `cardio.spo2` | **Watch only** | Only source that measures it. |
| `sleep.*` | **Ring only** | Only source with sleep tracking. |
| `load.*` | **Watch only** | Only source with strain / recovery. |

Every value carries its origin: `"source": {"rhr":"ring","hrv":"ring","spo2":"watch"}`.

---

## Cleaning decisions

| Decision | What I did | Why |
|---|---|---|
| **Timezone** | Parse every timestamp as an absolute UTC instant → convert to the patient's *real* zone (from `patients.csv`) → bucket to local calendar day | The ring stamps `+02:00` for every patient regardless of location. A naive `timestamp[:10]` misfiles a Tokyo patient's night by a full day. |
| **Sleep attribution** | Evening samples (local hour ≥ 18) roll forward to the wake date | Last night's recovery lands on the morning it informs — the Oura/Whoop convention. |
| **Dedup** | `drop_duplicates()` at load | Watch feed carries ~2% exact duplicate rows. Pipeline is a pure function of inputs → idempotent. |
| **Unit fix** | `spo2 × 100 if spo2 ≤ 1.5` | Watch sends SpO2 as a fraction (0.97). The guard prevents double-scaling if firmware is ever fixed upstream. |
| **Outliers** | Two stages: hard physiological bounds (`steps > 60k` → nulled) then robust 3× IQR per patient (→ flagged). Always recorded in `flags.outliers` with metric/value/method/action | Flag, never silently drop. An impossible value is nulled so it can't poison stats, but it stays visible and auditable. |
| **Missing data** | Interpolate sleep score across single-night gaps only, tagged `partial`. Gait, rHR, HRV, SpO2 **never** interpolated. Gaps > 1 day never filled | Fabricating a clinical signal to fill a gap would mislead a doctor. Sleep score is a smoothed index — one night is safely estimable. |
| **Disagreement** | Log `cardio.disagreementDelta["hrv"]` when ring and watch both reported | Ring reads ~20 ms higher than watch. Logged, not silently averaged — the conflict is information. |

---

## Data-quality model

Every section carries `dataQuality: "ok" | "partial" | "missing"`. **No silent NaNs** — pydantic rejects them at construction, so a null is always an explicit `None` with a traceable reason in `flags`.

**Two named edge cases, surfaced explicitly:**

```
Non-wear day  →  gait.dataQuality = "missing"
                 flags.missingSources = ["insole"]
                 gait is never borrowed from a wearable

Sensor drift  →  watch median HR > 150 detected
                 watch cardio + load discarded, cardio falls back to ring
                 flags.outliers = [{metric: "watch_sensor_drift",
                                    action: "watch_cardio_discarded_fell_back_to_ring"}]
```

Both appear in all 4 patients (4 non-wear days, 1 drift day each). Neither silently pollutes the summary.

---

## How I kept the LLM safe

```
summaries ──► _scrub() ──► _coverage() ──► LLM ──► _verify_grounding() ──► validated JSON
              strip PHI    cap confidence  JSON     check every number
```

1. **No raw PHI in the prompt** — no patientId, no names, no DOB. Only `age_band` (e.g. `adult_65_74`) plus numeric time-series. Reading dates stay: not PHI, and the doctor needs them for traceability.
2. **The model may abstain** — I compute coverage myself and *cap* the model's confidence. Under 25% field coverage it's forced to `"insufficient"` with zero findings, so it can't over-claim on thin data.
3. **Structured output** — JSON only, validated against a pydantic `NarrativeOutput` schema. Anything that doesn't parse is rejected, not shown.
4. **Grounding — no invented findings** — every `watchItem` carries `tracedTo {metric, value, date}`. After the model responds, each value is checked against the real summaries. Ungrounded claims are dropped before the doctor ever sees them.

Real example (P004): concern *"HRV lower at end of last week"* → `tracedTo: {hrv: 25.2, date: "2025-07-01"}` → matches exactly the value in that day's summary.

**Fallback chain:** `gemini-2.5-flash-lite` → free models → deterministic Python builder. The pipeline never blocks on LLM availability, and the engine used is tagged in the output.

---

## Validation

```bash
$ python run.py --check-idempotency
idempotency OK — summary hashes identical across two full runs:
  P001  adda0e5679d2eb827a061b337b2981b3    P003  7cbfca5f7472dc7aa1db705cbaf419d7
  P002  7e784b8db7b7ac55512cb58b82f7166a    P004  a55f614bc9ff2dc54dfa31d261abd5c2

$ pytest -v
test_timezone_bucketing_previous_local_day  PASSED   # UTC instant → correct local day
test_dedup_is_idempotent                    PASSED   # doubled rows → same aggregates
test_impossible_spike_flagged_not_dropped   PASSED   # 92k steps nulled AND visible in flags
3 passed
```

124 patient-days validated: `schema_version` on every row, `dataQuality` on every section, every non-null cardio value source-tagged.

---

## Scaling nightly to thousands of patients

The unit of work is a single `(patient, day)` with no cross-patient state, so the job is embarrassingly parallel: a scheduler fans out one task per patient shard across a worker pool, each worker reading only that patient's new raw partitions and recomputing just the affected days. Because the pipeline is idempotent and `computedAt` is pinned, retries and backfills are safe — no double-counting, no drift — and summaries upsert on `(patientId, date, schema_version)`, so a schema bump is a targeted rerun rather than a migration. The LLM narrative is the only external call, so it's rate-limited, cached per `(patient, 30-day window hash)`, and always degradable to the deterministic path if the provider is down.

---

## What works · what I'd improve next

**Works:** end-to-end from a clean clone in one command · timezone-correct bucketing across 4 zones · idempotent (hash-proven) · dedup, unit fix, two-stage outlier flagging · both named edge cases explicit in flags · cross-source disagreement logged · no silent NaNs · PHI-safe grounded narrative with abstention · 3 unit tests · bonus Streamlit dashboard.

**Next:** per-patient rolling-baseline model to catch *subtle* drift (my threshold only catches median HR > 150 — a slow drift from 65→95 bpm would slip through) · a real trend detector ("HRV down 4 days straight") feeding `watchItems` · a rubric-based eval harness scoring narrative groundedness · YAML config for thresholds instead of constants · property-based tests around DST boundaries · true sleep hours from the ring's hypnogram rather than sample-window span.

**Also honest:** sleep `hours` is a sample-span approximation, the IQR check needs ≥4 valid days to mean anything on sparse patients, and interpolation is deliberately minimal — which means more `partial` fields than a more aggressive pipeline would show.

---

## AI tools used — honestly

I used **Claude (Opus 4.8 / Sonnet 4.6)** as a pair-programmer: scaffolding files, drafting the generator, and debugging timezone-bucketing edge cases. Every design decision here — source priority, the wake-attribution rule, what to interpolate and what to refuse to interpolate, the two-stage outlier approach and the four narrative safety layers.

At runtime the narrative function calls **`google/gemini-2.5-flash-lite`** via **OpenRouter**.
