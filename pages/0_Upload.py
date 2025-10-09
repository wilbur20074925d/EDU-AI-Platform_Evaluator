import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Data", layout="wide")
st.title("Upload All Datasets (One Stop)")

st.markdown("""
Upload your six files here. They will be stored in `st.session_state` and used across pages:
- **Assessments** (pre/post)
- **Surveys**
- **Telemetry**
- **Teacher Logs**
- **Fairness**
- **Reflections**
""")

def load_any(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".jsonl") or file.name.endswith(".json"):
        return pd.read_json(file, lines=True)
    else:
        st.warning("Unsupported file type; use CSV or JSONL."); return None

# 1) Assessments
st.subheader("1) Assessments")
f_assess = st.file_uploader("Assessments CSV/JSONL", type=["csv","jsonl","json"], key="assess_up")
if f_assess:
    df = load_any(f_assess)
    st.write(df.head())
    st.session_state["assess_df"] = df

# 2) Surveys
st.subheader("2) Surveys")
f_survey = st.file_uploader("Surveys CSV/JSONL", type=["csv","jsonl","json"], key="survey_up")
if f_survey:
    df = load_any(f_survey)
    st.write(df.head())
    st.session_state["surveys_df"] = df

# 3) Telemetry
st.subheader("3) Telemetry")
f_tele = st.file_uploader("Telemetry CSV/JSONL", type=["csv","jsonl","json"], key="tele_up")
if f_tele:
    df = load_any(f_tele)
    st.write(df.head())
    st.session_state["tele_df"] = df

# 4) Teacher Logs
st.subheader("4) Teacher Logs")
f_teacher = st.file_uploader("Teacher Logs CSV/JSONL", type=["csv","jsonl","json"], key="teach_up")
if f_teacher:
    df = load_any(f_teacher)
    st.write(df.head())
    st.session_state["teacher_df"] = df

# 5) Fairness
st.subheader("5) Fairness / Compliance")
f_fair = st.file_uploader("Fairness CSV/JSONL", type=["csv","jsonl","json"], key="fair_up")
if f_fair:
    df = load_any(f_fair)
    st.write(df.head())
    st.session_state["fairness_df"] = df

# 6) Reflections
st.subheader("6) Reflections / Discourse")
f_refl = st.file_uploader("Reflections CSV/JSONL", type=["csv","jsonl","json"], key="refl_up")
if f_refl:
    df = load_any(f_refl)
    st.write(df.head())
    st.session_state["reflections_df"] = df

st.success("Uploads ready for the module pages once each file is provided.")