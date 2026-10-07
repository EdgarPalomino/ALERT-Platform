from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"

COMPLIANCE_LABELS = ["refusal", "partial", "full", "classifier_refused", "unknown"]
GUARD_LABELS = ["safe", "unsafe", "unknown"]

COMPLIANCE_COLORS = {
    "refusal": "#2ecc71",
    "partial": "#f39c12",
    "full": "#e74c3c",
    "classifier_refused": "#9b59b6",
    "unknown": "#95a5a6",
}

GUARD_COLORS = {
    "safe": "#2ecc71",
    "unsafe": "#e74c3c",
    "unknown": "#95a5a6",
}

MODEL_PALETTE = px.colors.qualitative.Set2


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: dict) -> list[dict[str, Any]]:
    """Flatten nested JSON into a list of per-attack dicts."""
    rows = []
    for record in data.get("results", []):
        record_id = record.get("record_index")
        base_prompt = record.get("record", {}).get("base_prompt", "")
        for attack in record.get("attacks", []):
            if not attack.get("success"):
                continue
            strategy = attack.get("attack", {}).get("strategy_name", "unknown")
            attacked_prompt = attack.get("attack", {}).get("attacked_prompt", "")
            response_text = attack.get("response", {}).get("text", "")
            latency = attack.get("response", {}).get("latency_seconds")
            prompt_tokens = attack.get("response", {}).get("prompt_tokens")
            completion_tokens = attack.get("response", {}).get("completion_tokens")
            compliance_label = (
                attack.get("classifications", {})
                .get("compliance", {})
                .get("label", "unknown")
            )
            guard_label = (
                attack.get("classifications", {})
                .get("llama_guard", {})
                .get("label", "unknown")
            )
            guard_category = (
                attack.get("classifications", {})
                .get("llama_guard", {})
                .get("metadata", {})
                .get("violation_category")
            )
            rows.append({
                "record_index": record_id,
                "base_prompt": base_prompt,
                "strategy": strategy,
                "attacked_prompt": attacked_prompt,
                "response": response_text,
                "latency_seconds": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "compliance": compliance_label,
                "llama_guard": guard_label,
                "guard_category": guard_category,
                "cross": f"{compliance_label} + {guard_label}",
            })
    return rows


def summary_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        "total_attacks": total,
        "refusal_rate": (df["compliance"] == "refusal").sum() / total if total else 0,
        "full_rate": (df["compliance"] == "full").sum() / total if total else 0,
        "partial_rate": (df["compliance"] == "partial").sum() / total if total else 0,
        "unsafe_rate": (df["llama_guard"] == "unsafe").sum() / total if total else 0,
        "jailbreak_rate": ((df["compliance"] == "full") & (df["llama_guard"] == "unsafe")).sum() / total if total else 0,
        "avg_latency": df["latency_seconds"].mean(),
    }


# ── Page ───────────────────────────────────────────────────────────────────────
st.title("📊 Analyze Results")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
available = sorted(OUTPUTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if not available:
    st.warning("No output files found in outputs/. Run the pipeline first.")
    st.stop()

available_map = {p.stem: str(p) for p in available}
default_selection = []
if "analyze_files" in st.session_state:
    for f in st.session_state["analyze_files"]:
        stem = Path(f).stem
        if stem in available_map:
            default_selection.append(stem)

selected_stems = st.multiselect(
    "Select runs to analyze",
    options=list(available_map.keys()),
    default=default_selection or [list(available_map.keys())[0]],
)

if not selected_stems:
    st.info("Select at least one run above.")
    st.stop()

# Load and build dataframes per run
run_dfs: dict[str, pd.DataFrame] = {}
run_summaries: dict[str, dict] = {}
for stem in selected_stems:
    data = load_json(available_map[stem])
    rows = extract_records(data)
    if rows:
        df = pd.DataFrame(rows)
        run_dfs[stem] = df
        run_summaries[stem] = summary_metrics(df)

if not run_dfs:
    st.warning("No successful attack results found in the selected runs.")
    st.stop()

multi = len(run_dfs) > 1
color_map = {stem: MODEL_PALETTE[i % len(MODEL_PALETTE)] for i, stem in enumerate(run_dfs)}

st.divider()

# ── Summary cards ──────────────────────────────────────────────────────────────
st.subheader("Summary")
for stem, metrics in run_summaries.items():
    label = f"**{stem}**" if multi else ""
    if multi:
        st.markdown(label)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Attacks", metrics["total_attacks"])
    col2.metric("Refusal Rate", f"{metrics['refusal_rate']:.1%}")
    col3.metric("Partial Rate", f"{metrics['partial_rate']:.1%}")
    col4.metric("Full Compliance", f"{metrics['full_rate']:.1%}")
    col5.metric("Unsafe (Guard)", f"{metrics['unsafe_rate']:.1%}")
    col6.metric("🚨 Jailbreak Rate", f"{metrics['jailbreak_rate']:.1%}")
    if multi:
        st.divider()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Attack Breakdown", "🔀 Cross-Reference", "⏱️ Latency & Performance", "🔍 Record Explorer"])

# ── Tab 1: Attack breakdown ────────────────────────────────────────────────────
with tab1:
    st.subheader("Compliance by Attack Strategy")
    frames = []
    for stem, df in run_dfs.items():
        grp = df.groupby(["strategy", "compliance"]).size().reset_index(name="count")
        grp["run"] = stem
        frames.append(grp)
    combined = pd.concat(frames)

    if multi:
        fig = px.bar(
            combined, x="strategy", y="count", color="run",
            facet_col="compliance", barmode="group",
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            combined, x="strategy", y="count", color="compliance",
            barmode="group", color_discrete_map=COMPLIANCE_COLORS,
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("LlamaGuard by Attack Strategy")
    frames = []
    for stem, df in run_dfs.items():
        grp = df.groupby(["strategy", "llama_guard"]).size().reset_index(name="count")
        grp["run"] = stem
        frames.append(grp)
    combined = pd.concat(frames)

    if multi:
        fig = px.bar(
            combined, x="strategy", y="count", color="run",
            facet_col="llama_guard", barmode="group",
            color_discrete_map=color_map,
        )
    else:
        fig = px.bar(
            combined, x="strategy", y="count", color="llama_guard",
            barmode="group", color_discrete_map=GUARD_COLORS,
        )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Cross-reference ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Compliance × LlamaGuard Heatmap")
    for stem, df in run_dfs.items():
        if multi:
            st.markdown(f"**{stem}**")
        pivot = df.groupby(["compliance", "llama_guard"]).size().unstack(fill_value=0)
        fig = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="Reds",
            labels={"x": "LlamaGuard", "y": "Compliance", "color": "Count"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-Reference Distribution")
    frames = []
    for stem, df in run_dfs.items():
        grp = df.groupby("cross").size().reset_index(name="count")
        grp["run"] = stem
        frames.append(grp)
    combined = pd.concat(frames)

    fig = px.bar(
        combined, x="cross", y="count",
        color="run" if multi else "cross",
        barmode="group",
        color_discrete_map=color_map if multi else None,
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Latency & performance ───────────────────────────────────────────────
with tab3:
    st.subheader("Latency Distribution")
    frames = []
    for stem, df in run_dfs.items():
        tmp = df[["latency_seconds"]].copy()
        tmp["run"] = stem
        frames.append(tmp)
    combined = pd.concat(frames)

    fig = px.histogram(
        combined, x="latency_seconds", color="run" if multi else None,
        nbins=30, barmode="overlay", opacity=0.75,
        color_discrete_map=color_map if multi else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency by Attack Strategy")
    frames = []
    for stem, df in run_dfs.items():
        tmp = df[["strategy", "latency_seconds"]].copy()
        tmp["run"] = stem
        frames.append(tmp)
    combined = pd.concat(frames)

    fig = px.box(
        combined, x="strategy", y="latency_seconds",
        color="run" if multi else "strategy",
        color_discrete_map=color_map if multi else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Token Counts")
    frames = []
    for stem, df in run_dfs.items():
        tmp = df[["prompt_tokens", "completion_tokens"]].copy()
        tmp["run"] = stem
        frames.append(tmp)
    combined = pd.concat(frames)

    fig = px.scatter(
        combined, x="prompt_tokens", y="completion_tokens",
        color="run" if multi else None,
        opacity=0.6,
        color_discrete_map=color_map if multi else None,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Record explorer ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Per-Record Explorer")

    explorer_run = st.selectbox("Run", options=list(run_dfs.keys())) if multi else list(run_dfs.keys())[0]
    df = run_dfs[explorer_run]

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_compliance = st.multiselect("Compliance", options=df["compliance"].unique().tolist(), default=df["compliance"].unique().tolist())
    with col2:
        filter_guard = st.multiselect("LlamaGuard", options=df["llama_guard"].unique().tolist(), default=df["llama_guard"].unique().tolist())
    with col3:
        filter_strategy = st.multiselect("Strategy", options=df["strategy"].unique().tolist(), default=df["strategy"].unique().tolist())

    filtered = df[
        df["compliance"].isin(filter_compliance) &
        df["llama_guard"].isin(filter_guard) &
        df["strategy"].isin(filter_strategy)
    ].reset_index(drop=True)

    st.caption(f"Showing {len(filtered)} of {len(df)} results")

    for _, row in filtered.iterrows():
        with st.expander(f"[{row['strategy']}] Record {row['record_index']} — compliance: `{row['compliance']}` | guard: `{row['llama_guard']}`"):
            st.markdown("**Base Prompt**")
            st.text(row["base_prompt"])
            st.markdown("**Attacked Prompt**")
            st.text(row["attacked_prompt"])
            st.markdown("**Model Response**")
            st.text(row["response"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Latency", f"{row['latency_seconds']:.2f}s")
            col2.metric("Prompt Tokens", row["prompt_tokens"])
            col3.metric("Completion Tokens", row["completion_tokens"])
