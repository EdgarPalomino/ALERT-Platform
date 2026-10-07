from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
PIPELINE_SCRIPT = REPO_ROOT / "src" / "run_pipeline.py"

OLLAMA_MODELS = [
    "llama3.2:latest",
    "llama3.2-3b",
    "deepseek-r1:1.5b",
    "mistral:7b",
    "qwen2.5:7b",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

# ── Redirect hook (must be before any widget renders) ──────────────────────────
if st.session_state.get("switch_to_analyze"):
    st.session_state["switch_to_analyze"] = False
    st.switch_page("pages/analyze.py")

st.title("🚀 Run Pipeline")
st.caption("Configure and launch a red-teaming experiment.")

# ── Recent runs ────────────────────────────────────────────────────────────────
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
recent_runs = sorted(OUTPUTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

if recent_runs:
    st.subheader("Recent Runs")
    cols = st.columns(min(len(recent_runs), 4))
    for i, run_file in enumerate(recent_runs[:4]):
        with cols[i]:
            if st.button(f"📄 {run_file.stem}", use_container_width=True):
                st.session_state["analyze_files"] = [str(run_file)]
                st.switch_page("pages/analyze.py")
    st.divider()

# ── Backend selector (outside form so it reruns on change) ────────────────────
st.subheader("Target")
backend = st.radio("Backend", ["ollama", "openai_api"], horizontal=True)

# ── Config form ────────────────────────────────────────────────────────────────
with st.form("run_form"):
    st.subheader("Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        data_path = st.text_input("Data path", value="data/advbench/train-00000-of-00001.parquet")
    with col2:
        limit = st.number_input("Limit (prompts)", min_value=1, max_value=500, value=10)
    with col3:
        start_index = st.number_input("Start index", min_value=0, value=0)

    st.subheader("Attack")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_identity = st.checkbox("Identity", value=True)
    with col2:
        use_roleplay = st.checkbox("Roleplay", value=True)
    with col3:
        use_obfuscation = st.checkbox("Obfuscation", value=True)
    with col4:
        use_prefix = st.checkbox("Prefix Injection", value=True)

    prefix_mode = st.selectbox(
        "Prefix mode",
        ["no_safety_or_refusal", "direct_answer", "no_safety_commentary", "maximal_attack"],
    )

    # Target section inside form (no backend radio here anymore)
    col1, col2 = st.columns(2)
    if backend == "ollama":
        with col1:
            ollama_model = st.selectbox("Model", OLLAMA_MODELS)
            custom_ollama = st.text_input("Or enter custom model name", placeholder="e.g. llama3.1:8b")
            target_model = custom_ollama if custom_ollama else ollama_model
        with col2:
            ollama_base_url = st.text_input("Ollama base URL", value="http://localhost:11434")
            ollama_num_ctx = st.number_input("Context window (num_ctx)", value=4096, step=512)
    else:
        with col1:
            openai_model = st.selectbox("Model", OPENAI_MODELS)
            custom_openai = st.text_input("Or enter custom model name", placeholder="e.g. gpt-4-turbo")
            target_model = custom_openai if custom_openai else openai_model
        with col2:
            openai_base_url = st.text_input("Base URL", value="https://api.openai.com/v1")
            openai_api_key = st.text_input("API key", type="password", placeholder="sk-...")

    st.subheader("Classifiers")
    col1, col2 = st.columns(2)
    with col1:
        use_compliance = st.checkbox("Compliance classifier", value=True)
        compliance_model = st.text_input("Compliance model", value="llama3.2:latest")
    with col2:
        use_llama_guard = st.checkbox("LlamaGuard classifier", value=True)
        llama_guard_model = st.text_input("LlamaGuard model", value="llama-guard3:1b")

    st.subheader("Output")
    col1, col2 = st.columns(2)
    with col1:
        run_name = st.text_input("Run name", value="my_experiment")
    with col2:
        save_pretty = st.checkbox("Pretty JSON", value=True)

    submitted = st.form_submit_button("🚀 Launch Run", use_container_width=True, type="primary")

# ── Build and execute command ──────────────────────────────────────────────────
if submitted:
    cmd = [
        sys.executable, str(PIPELINE_SCRIPT),
        "--dataset.data_path", data_path,
        "--dataset.limit", str(limit),
        "--dataset.start_index", str(start_index),
        "--attack.use_identity", str(use_identity).lower(),
        "--attack.use_roleplay", str(use_roleplay).lower(),
        "--attack.use_obfuscation", str(use_obfuscation).lower(),
        "--attack.use_prefix_injection", str(use_prefix).lower(),
        "--attack.prefix_mode", prefix_mode,
        "--target.backend", backend,
        "--target.model_name", target_model,
        "--classifier.use_compliance", str(use_compliance).lower(),
        "--classifier.compliance_model", compliance_model,
        "--classifier.use_llama_guard", str(use_llama_guard).lower(),
        "--classifier.llama_guard_model", llama_guard_model,
        "--output.run_name", run_name,
        "--output.save_pretty_json", str(save_pretty).lower(),
    ]

    if backend == "ollama":
        cmd += [
            "--target.ollama_base_url", ollama_base_url,
            "--target.ollama_num_ctx", str(ollama_num_ctx),
        ]
    else:
        cmd += [
            "--target.base_url", openai_base_url,
            "--target.api_key", openai_api_key,
        ]

    st.subheader("Live Output")
    log_box = st.empty()
    status = st.empty()

    log_lines = []
    start = time.perf_counter()

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    ) as proc:
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            log_box.code("\n".join(log_lines[-40:]), language="bash")

        proc.wait()
        elapsed = time.perf_counter() - start

    if proc.returncode == 0:
        output_file = OUTPUTS_DIR / f"{run_name}.json"
        status.success(f"✅ Run complete in {elapsed:.1f}s — saved to outputs/{run_name}.json")
        if output_file.exists():
            st.session_state["analyze_files"] = [str(output_file)]
            st.session_state["run_complete"] = True
    else:
        status.error(f"❌ Run failed after {elapsed:.1f}s — check the log above for errors")

# ── View results button (persists after run completes) ─────────────────────────
if st.session_state.get("run_complete"):
    if st.button("📊 View Results in Analyze", type="primary", use_container_width=True):
        st.session_state["run_complete"] = False
        st.session_state["switch_to_analyze"] = True
        st.rerun()
