import streamlit as st

st.set_page_config(
    page_title="LLM Red Team Bench",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

run_page = st.Page("pages/run.py", title="Run", icon="🚀")
analyze_page = st.Page("pages/analyze.py", title="Analyze", icon="📊")

pg = st.navigation([run_page, analyze_page])
pg.run()
