"""
dashboard.py  —  Bonus: Streamlit patient dashboard
=====================================================
Shows a patient's 30-day summary + AI narrative in a clean clinical view.

Run:
    streamlit run dashboard.py

Requires outputs/ to exist (run `python run.py` first).
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
SUMMARIES_DIR = Path("outputs/summaries")
NARRATIVES_DIR = Path("outputs/narratives")

PATIENT_LABELS = {
    "P001": "P001 — Paris (adult 35–44)",
    "P002": "P002 — New York (adult 55–64)",
    "P003": "P003 — Tokyo (adult 25–34)",
    "P004": "P004 — London (adult 65–74)",
}

QUALITY_COLOR = {"ok": "ok", "partial": "partial", "missing": "missing"}
CONF_COLOR    = {"high": "high", "medium": "medium", "low": "low", "insufficient": "insufficient"}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="eSteps Health — Patient Dashboard",
    page_icon="👟",
    layout="wide",
)

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    border-left: 4px solid #0F6E56;
}
.flag-card {
    background: #fff8e1 !important;
    border-radius: 8px;
    padding: 10px 14px;
    border-left: 4px solid #f9a825;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #1a0800 !important;
}
.drift-card {
    background: #fce4ec !important;
    border-radius: 8px;
    padding: 10px 14px;
    border-left: 4px solid #c62828;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #3b0000 !important;
}
.narrative-box {
    background: #e8f5e9 !important;
    border-radius: 10px;
    padding: 18px 22px;
    border-left: 5px solid #2e7d32;
    margin-bottom: 12px;
    color: #0d2b10 !important;
}
.watch-item {
    background: #fff8e1;
    border-radius: 8px;
    padding: 14px 18px;
    border: 1px solid #ffcc80;
    border-left: 5px solid #e65100;
    margin-bottom: 10px;
    font-size: 0.92rem;
    color: #3e2000;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.source-tag {
    background: #e3f2fd;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.78rem;
    color: #1565c0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_patient(pid):
    summary_path  = SUMMARIES_DIR / f"{pid}.json"
    narrative_path = NARRATIVES_DIR / f"{pid}.json"
    if not summary_path.exists():
        return None, None
    summaries = json.load(open(summary_path))
    narrative = json.load(open(narrative_path)) if narrative_path.exists() else {}
    return summaries, narrative

def to_df(summaries):
    rows = []
    for d in summaries:
        rows.append({
            "date":      d["date"],
            "steps":     d["gait"]["steps"],
            "cadence":   d["gait"]["cadence"],
            "symmetry":  d["gait"]["symmetry"],
            "gait_q":    d["gait"]["dataQuality"],
            "rhr":       d["cardio"]["rhr"],
            "hrv":       d["cardio"]["hrv"],
            "spo2":      d["cardio"]["spo2"],
            "cardio_q":  d["cardio"]["dataQuality"],
            "hrv_delta": d["cardio"]["disagreementDelta"].get("hrv"),
            "rhr_src":   d["cardio"]["source"].get("rhr", "—"),
            "hrv_src":   d["cardio"]["source"].get("hrv", "—"),
            "sleep_score": d["sleep"]["score"],
            "sleep_hours": d["sleep"]["hours"],
            "sleep_q":   d["sleep"]["dataQuality"],
            "recovery":  d["load"]["recovery"],
            "strain":    d["load"]["strain"],
            "load_q":    d["load"]["dataQuality"],
            "n_outliers": len(d["flags"]["outliers"]),
            "drift":     any(o["metric"] == "watch_sensor_drift"
                             for o in d["flags"]["outliers"]),
            "nonwear":   "insole" in d["flags"]["missingSources"],
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👟 eSteps Health — Patient Dashboard")
st.caption("Track 2 Data Challenge · multi-source sensor fusion + AI narrative")

# ── Sidebar: patient selector ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient")
    available = sorted([p.stem for p in SUMMARIES_DIR.glob("*.json")]) if SUMMARIES_DIR.exists() else []
    if not available:
        st.error("No summaries found. Run `python run.py` first.")
        st.stop()

    pid = st.selectbox("Select patient", available,
                       format_func=lambda x: PATIENT_LABELS.get(x, x))

    summaries, narrative = load_patient(pid)
    if summaries is None:
        st.error("Could not load data.")
        st.stop()

    df = to_df(summaries)

    st.divider()
    st.caption(f"**{len(df)} days** · {df.date.min().strftime('%b %d')} – {df.date.max().strftime('%b %d, %Y')}")

    nonwear_days = df[df.nonwear].shape[0]
    drift_days   = df[df.drift].shape[0]
    st.metric("Non-wear days",   nonwear_days)
    st.metric("Sensor-drift days", drift_days)
    st.metric("Gait OK days",    df[df.gait_q == "ok"].shape[0])

    st.divider()
    selected_date = st.selectbox(
        "Inspect a specific day",
        df["date"].dt.strftime("%Y-%m-%d").tolist(),
        index=len(df) - 1,
    )

# ── AI Narrative ──────────────────────────────────────────────────────────────
st.subheader("🧠 AI Clinical Narrative")
if narrative:
    conf = narrative.get("dataConfidence", "—")
    engine = narrative.get("engine", "—")
    model  = narrative.get("model_used", "deterministic_fallback")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<div class="narrative-box">{narrative.get("summary","—")}</div>',
            unsafe_allow_html=True)
    with col2:
        conf_bg = {"high":"#c8e6c9","medium":"#fff9c4","low":"#ffe0b2","insufficient":"#ffcdd2"}.get(conf,"#f5f5f5")
        conf_fg = {"high":"#1b5e20","medium":"#e65100","low":"#bf360c","insufficient":"#b71c1c"}.get(conf,"#333")
        conf_solid = {"high":"#2e7d32","medium":"#e65100","low":"#bf360c","insufficient":"#c62828"}.get(conf,"#555")
        st.markdown(f"**Confidence:** &nbsp;<span style='background:{conf_solid};color:#ffffff;padding:3px 12px;border-radius:12px;font-weight:700;font-size:0.85rem;letter-spacing:0.4px;'>{conf.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Engine:** `{engine}`")
        st.markdown(f"**Model:** `{model}`")

    if narrative.get("watchItems"):
        st.markdown("**Things to watch:**")
        for item in narrative["watchItems"]:
            t = item["tracedTo"]
            st.markdown(
                f'<div class="watch-item">'
                f'<div style="font-weight:700;font-size:0.95rem;color:#1a0800;margin-bottom:6px;">#{item["rank"]}  {item["concern"]}</div>'
                f'<div style="font-size:0.82rem;color:#ffffff;font-family:monospace;background:#bf360c;padding:4px 10px;border-radius:4px;display:inline-block;font-weight:600;">'
                f'metric: {t["metric"]}  |  value: {t["value"]}  |  date: {t["date"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.info("No specific concerns flagged for this patient.")
else:
    st.warning("No narrative found.")

st.divider()

# ── 30-day trend charts ───────────────────────────────────────────────────────
st.subheader("📈 30-day trends")

tab1, tab2, tab3, tab4 = st.tabs(["Gait", "Cardio", "Sleep", "Load"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily Steps**")
        chart_df = df[["date","steps"]].set_index("date")
        # mark non-wear and spike days
        nonwear_mask = df.nonwear
        drift_mask   = df.drift
        st.line_chart(chart_df, use_container_width=True)
        if nonwear_mask.any():
            st.caption(f"⬤ Non-wear gaps: {', '.join(df[nonwear_mask].date.dt.strftime('%b %d').tolist())}")
    with c2:
        st.markdown("**Gait Symmetry** (1.0 = perfect)")
        st.line_chart(df[["date","symmetry"]].set_index("date"), use_container_width=True)

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Resting HR (bpm)**")
        st.line_chart(df[["date","rhr"]].set_index("date"), use_container_width=True)
    with c2:
        st.markdown("**HRV (ms)**")
        st.line_chart(df[["date","hrv"]].set_index("date"), use_container_width=True)
        # HRV disagreement delta
        if df["hrv_delta"].notna().any():
            st.caption("Ring–Watch HRV delta (ms)")
            st.line_chart(df[["date","hrv_delta"]].set_index("date"), use_container_width=True)
    with c3:
        st.markdown("**SpO2 (%)**")
        st.line_chart(df[["date","spo2"]].set_index("date"), use_container_width=True)
    if drift_days:
        st.warning(f"⚠️ Sensor-drift detected on: "
                   f"{', '.join(df[df.drift].date.dt.strftime('%b %d').tolist())} "
                   f"— watch cardio discarded, fell back to ring on those days.")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sleep Score**")
        st.line_chart(df[["date","sleep_score"]].set_index("date"), use_container_width=True)
    with c2:
        st.markdown("**Sleep Hours**")
        st.line_chart(df[["date","sleep_hours"]].set_index("date"), use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Recovery (%)**")
        st.line_chart(df[["date","recovery"]].set_index("date"), use_container_width=True)
    with c2:
        st.markdown("**Strain**")
        st.line_chart(df[["date","strain"]].set_index("date"), use_container_width=True)

st.divider()

# ── Day inspector ─────────────────────────────────────────────────────────────
st.subheader(f"🔍 Day inspector — {selected_date}")

day = df[df["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]
raw_day = next(d for d in summaries if d["date"] == selected_date)

# flags first
if day["nonwear"]:
    st.markdown('<div class="flag-card">⬤ <b>Non-wear day</b> — no insole data. '
                'Gait is <code>missing</code>. Never borrowed from a wearable.</div>',
                unsafe_allow_html=True)
if day["drift"]:
    st.markdown('<div class="drift-card">🚨 <b>Sensor-drift day</b> — watch median HR exceeded 150 bpm. '
                'Watch cardio discarded. Cardio fell back to ring.</div>',
                unsafe_allow_html=True)
for o in raw_day["flags"]["outliers"]:
    if o["metric"] in ("steps","cadence"):
        st.markdown(f'<div class="flag-card">⚠️ <b>Outlier:</b> <code>{o["metric"]}</code> = '
                    f'{o["value"]} — {o["method"]} → {o["action"]}</div>',
                    unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**🦶 Gait**")
    q = day["gait_q"]
    qc = {"ok":"#c8e6c9","partial":"#fff9c4","missing":"#ffcdd2"}.get(q,"#eee")
    qt = {"ok":"#1b5e20","partial":"#e65100","missing":"#b71c1c"}.get(q,"#333")
    st.markdown(f'<span style="background:{qc};color:{qt};padding:2px 10px;border-radius:10px;font-size:0.82rem;font-weight:700;">{q.upper()}</span>', unsafe_allow_html=True)
    st.metric("Steps",    int(day["steps"]) if pd.notna(day["steps"]) else "—")
    st.metric("Cadence",  f'{day["cadence"]:.1f}' if pd.notna(day["cadence"]) else "—")
    st.metric("Symmetry", f'{day["symmetry"]:.3f}' if pd.notna(day["symmetry"]) else "—")

with col2:
    st.markdown("**❤️ Cardio**")
    q = day["cardio_q"]
    qc = {"ok":"#c8e6c9","partial":"#fff9c4","missing":"#ffcdd2"}.get(q,"#eee")
    qt = {"ok":"#1b5e20","partial":"#e65100","missing":"#b71c1c"}.get(q,"#333")
    st.markdown(f'<span style="background:{qc};color:{qt};padding:2px 10px;border-radius:10px;font-size:0.82rem;font-weight:700;">{q.upper()}</span>', unsafe_allow_html=True)
    rhr_src = f'<span class="source-tag">{day["rhr_src"]}</span>'
    hrv_src = f'<span class="source-tag">{day["hrv_src"]}</span>'
    st.metric("Resting HR", f'{day["rhr"]:.1f} bpm' if pd.notna(day["rhr"]) else "—")
    st.markdown(f"source: {rhr_src}", unsafe_allow_html=True)
    st.metric("HRV",  f'{day["hrv"]:.1f} ms' if pd.notna(day["hrv"]) else "—")
    st.markdown(f"source: {hrv_src}", unsafe_allow_html=True)
    st.metric("SpO2", f'{day["spo2"]:.1f} %' if pd.notna(day["spo2"]) else "—")
    if pd.notna(day["hrv_delta"]):
        st.caption(f"Ring–Watch HRV delta: {day['hrv_delta']:+.1f} ms")

with col3:
    st.markdown("**😴 Sleep**")
    q = day["sleep_q"]
    qc = {"ok":"#c8e6c9","partial":"#fff9c4","missing":"#ffcdd2"}.get(q,"#eee")
    qt = {"ok":"#1b5e20","partial":"#e65100","missing":"#b71c1c"}.get(q,"#333")
    st.markdown(f'<span style="background:{qc};color:{qt};padding:2px 10px;border-radius:10px;font-size:0.82rem;font-weight:700;">{q.upper()}</span>', unsafe_allow_html=True)
    src = raw_day["sleep"].get("source","—")
    st.metric("Score", f'{day["sleep_score"]:.0f}' if pd.notna(day["sleep_score"]) else "—")
    st.metric("Hours", f'{day["sleep_hours"]:.1f} h' if pd.notna(day["sleep_hours"]) else "—")
    st.markdown(f'source: <span class="source-tag">{src}</span>', unsafe_allow_html=True)

with col4:
    st.markdown("**⚡ Load**")
    q = day["load_q"]
    qc = {"ok":"#c8e6c9","partial":"#fff9c4","missing":"#ffcdd2"}.get(q,"#eee")
    qt = {"ok":"#1b5e20","partial":"#e65100","missing":"#b71c1c"}.get(q,"#333")
    st.markdown(f'<span style="background:{qc};color:{qt};padding:2px 10px;border-radius:10px;font-size:0.82rem;font-weight:700;">{q.upper()}</span>', unsafe_allow_html=True)
    src = raw_day["load"].get("source","—")
    st.metric("Recovery", f'{day["recovery"]:.0f} %' if pd.notna(day["recovery"]) else "—")
    st.metric("Strain",   f'{day["strain"]:.1f}' if pd.notna(day["strain"]) else "—")
    st.markdown(f'source: <span class="source-tag">{src}</span>', unsafe_allow_html=True)

# raw JSON expander
with st.expander("Raw PatientDailySummary JSON"):
    st.json(raw_day)

st.divider()
st.caption("eSteps Health · Track 2 Data Challenge · pipeline v1.0")