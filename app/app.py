"""Modern department-based university inquiry operations dashboard."""

from html import escape

import pandas as pd
import streamlit as st

from db_utils import fetch_all, init_db, insert_inquiry, update_queue_item
from keywords import find_trigger_keywords
from model_utils import classify_inquiry, load_all_models

st.set_page_config(page_title="UniFlow | Inquiry Operations", page_icon="🎓", layout="wide")
init_db()

URGENCY_RANK = {"High": 0, "Medium": 1, "Low": 2}
URGENCY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
STATUSES = ["New", "In progress", "Waiting", "Resolved"]
ASSIGNEES = ["Unassigned", "Department officer", "Senior officer", "Department head"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
:root { --ink:#163c2d; --muted:#587568; --green:#16835f; --line:#dcebe3; }
.stApp { background:#f3f8f5; color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em!important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0c3b2e,#155843); border-right:0; }
[data-testid="stSidebar"] * { color:#e9f8f0!important; }
[data-testid="stSidebar"] [data-baseweb="radio"] label { padding:.55rem .75rem; border-radius:10px; }
[data-testid="stSidebar"] [aria-checked="true"] { background:#ffffff18; }
[data-testid="stMetric"] { position:relative; overflow:hidden; min-height:8.2rem; background:linear-gradient(145deg,#ffffff 20%,#f0faf5 100%); border:1px solid #d5eadf; padding:1.25rem 1.35rem; border-radius:18px; box-shadow:0 8px 24px #175c3c0d; transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease; }
[data-testid="stMetric"]::before { content:""; position:absolute; left:0; top:0; width:100%; height:4px; background:linear-gradient(90deg,#16835f,#4fc391); }
[data-testid="stMetric"]::after { content:""; position:absolute; width:5rem; height:5rem; right:-2rem; top:-2.2rem; border-radius:50%; background:#4fc39112; pointer-events:none; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:#b9dcca; box-shadow:0 14px 30px #175c3c18; }
[data-testid="stMetric"] * { opacity:1!important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * { color:#587568!important; font-size:.76rem!important; font-weight:700!important; letter-spacing:.075em!important; text-transform:uppercase!important; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * { color:#123d2c!important; font-family:'Manrope',sans-serif!important; font-size:clamp(1.35rem,2.1vw,2.05rem)!important; font-weight:800!important; letter-spacing:-.045em!important; line-height:1.2!important; }
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] { width:fit-content; margin-top:.3rem; padding:.18rem .48rem; border-radius:999px; background:#e1f5ea; }
[data-testid="stMetricDelta"] * { color:#16835f!important; font-weight:700!important; }
.hero { background:linear-gradient(120deg,#0e4936,#16835f 70%,#32a779); padding:2rem 2.2rem; border-radius:22px; color:white; margin-bottom:1.2rem; box-shadow:0 16px 40px #1264472e; }
.hero h1 { margin:0; font-size:2rem; color:white; }.hero p { opacity:.82; margin:.45rem 0 0; }
.eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; opacity:.75; }
.ticket { background:white; border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; margin:.55rem 0; box-shadow:0 5px 18px #175c3c0b; }
.ticket-top { display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem; }
.ticket-text { margin:.65rem 0; font-weight:600; color:var(--ink); }
.pill { display:inline-block; padding:.25rem .55rem; border-radius:999px; background:#e8f7ef; color:#147052; font-size:.75rem; margin-right:.3rem; }
.stButton>button { border-radius:10px; font-weight:700; }
.stButton>button:not([kind="primary"]) { background:white!important; color:#147052!important; border:1px solid #b9dcca!important; box-shadow:0 3px 9px #175c3c0a!important; }
.stButton>button:not([kind="primary"]):hover { background:#e8f7ef!important; color:#0f6046!important; border-color:#16835f!important; }
.stButton>button[kind="primary"] { background:#16835f; border-color:#16835f; }
.stButton>button[kind="primary"]:hover { background:#0f6b4c; border-color:#0f6b4c; }
[data-testid="baseButton-secondary"] { background:white!important; color:#147052!important; border:1px solid #b9dcca!important; border-radius:10px!important; box-shadow:0 3px 9px #175c3c0a!important; }
[data-testid="baseButton-secondary"] * { color:#147052!important; }
[data-testid="baseButton-secondary"]:hover { background:#e8f7ef!important; border-color:#16835f!important; }
[data-testid="baseButton-primary"] { background:#16835f!important; color:white!important; border-color:#16835f!important; }
[data-testid="baseButton-primary"] * { color:white!important; }
.stAlert { border-color:#cce7d8; }
.stTextArea textarea,.stSelectbox>div>div,.stNumberInput input { border-radius:11px!important; }
[data-testid="stExpander"] { background:white!important; border:1px solid var(--line)!important; border-radius:13px!important; overflow:hidden; box-shadow:0 4px 14px #175c3c08; }
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary { background:white!important; color:#163c2d!important; }
[data-testid="stExpander"] summary:hover { background:#e8f7ef!important; color:#147052!important; }
[data-testid="stExpander"] summary * { color:#163c2d!important; fill:#163c2d!important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { background:#fbfefc!important; border-top:1px solid var(--line); padding-top:1rem; }
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label * { color:#365f4f!important; font-weight:600!important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:white!important; color:#163c2d!important; border-color:#b9dcca!important; min-height:2.75rem; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { color:#163c2d!important; }
[data-baseweb="popover"] [role="listbox"] { background:white!important; border:1px solid var(--line)!important; border-radius:11px!important; box-shadow:0 12px 28px #123d2c24!important; }
[data-baseweb="popover"] [role="option"] { background:white!important; color:#163c2d!important; }
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] { background:#e8f7ef!important; color:#147052!important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.45rem; background:white; border:1px solid var(--line); border-radius:14px; padding:.4rem; overflow-x:auto; }
[data-testid="stTabs"] [data-baseweb="tab"] { height:2.65rem; padding:0 1rem; border-radius:10px; color:#587568; font-weight:700; white-space:nowrap; }
[data-testid="stTabs"] [aria-selected="true"] { background:#16835f!important; color:white!important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top:1.25rem; }
[data-testid="stButtonGroup"] { background:transparent!important; border:0!important; padding:0!important; min-height:0!important; }
[data-testid="stButtonGroup"] [data-baseweb="button-group"] { display:inline-flex!important; width:auto!important; min-height:0!important; gap:.3rem!important; background:white; border:1px solid var(--line); border-radius:12px; padding:.3rem; }
[data-testid="stButtonGroup"] [data-baseweb="button-group"] button { height:2.35rem!important; min-height:2.35rem!important; padding:0 1rem!important; margin:0!important; border:0!important; border-radius:8px!important; color:#587568!important; font-weight:700!important; background:transparent!important; box-shadow:none!important; }
[data-testid="stButtonGroup"] [data-baseweb="button-group"] button:hover { background:#e8f7ef!important; color:#147052!important; }
[data-testid="stButtonGroup"] [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] { background:#16835f!important; color:white!important; box-shadow:0 4px 10px #12644726!important; }
[data-testid="stButtonGroup"] [data-baseweb="button-group"] button[role="radio"][aria-checked="true"] * { color:white!important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing the AI routing engine…")
def get_models():
    return load_all_models()

try:
    models = get_models()
except FileNotFoundError as exc:
    st.error(str(exc)); st.stop()

departments = list(models["category"][2])
with st.sidebar:
    st.markdown("## ◈ UniFlow")
    st.caption("INQUIRY OPERATIONS")
    st.markdown("---")
    page = st.radio("Workspace", ["Overview", "New inquiry", "Department queues"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("AI ROUTING ENGINE")
    st.success("● Online")

df = fetch_all()
if not df.empty:
    df["department"] = df["department"].fillna(df["predicted_category"])
    df["queue_status"] = df["queue_status"].fillna("New")
    df["assignee"] = df["assignee"].fillna("Unassigned")

def hero(title, subtitle, eyebrow="OPERATIONS CENTER"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def render_tickets(data):
    if data.empty:
        st.info("This queue is clear. New inquiries routed here will appear automatically.")
        return
    data = data.assign(rank=data["predicted_urgency"].map(URGENCY_RANK)).sort_values(["rank", "id"])
    for _, row in data.iterrows():
        text = escape(str(row["raw_text"]))
        department = escape(str(row["department"]))
        queue_status = escape(str(row["queue_status"]))
        assignee = escape(str(row["assignee"]))
        st.markdown(f'''<div class="ticket"><div class="ticket-top"><span>INQ-{int(row['id']):04d} · {row['created_at']}</span><span>{URGENCY_ICON.get(row['predicted_urgency'],'')} {row['predicted_urgency']} priority</span></div><div class="ticket-text">{text}</div><span class="pill">{department}</span><span class="pill">{queue_status}</span><span class="pill">{assignee}</span></div>''', unsafe_allow_html=True)
        with st.expander(f"Manage INQ-{int(row['id']):04d}"):
            c1, c2, c3 = st.columns([2,2,1], vertical_alignment="bottom")
            status = c1.selectbox("Status", STATUSES, index=STATUSES.index(row["queue_status"]) if row["queue_status"] in STATUSES else 0, key=f"s{row['id']}")
            owner = c2.selectbox("Owner", ASSIGNEES, index=ASSIGNEES.index(row["assignee"]) if row["assignee"] in ASSIGNEES else 0, key=f"a{row['id']}")
            if c3.button("Update", key=f"u{row['id']}", use_container_width=True):
                update_queue_item(int(row["id"]), status, owner); st.toast("Inquiry updated", icon="✅"); st.rerun()

if page == "Overview":
    hero("Good day, operations team", "A live view of inquiries across every university department.")
    total = len(df); open_count = len(df[df["queue_status"] != "Resolved"]) if total else 0
    high = len(df[(df["predicted_urgency"] == "High") & (df["queue_status"] != "Resolved")]) if total else 0
    resolved = len(df[df["queue_status"] == "Resolved"]) if total else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total inquiries", total); c2.metric("Open queue", open_count); c3.metric("High priority", high); c4.metric("Resolved", resolved)
    st.markdown("### Department activity")
    if total:
        activity = df.groupby("department").agg(Inquiries=("id","count"), Open=("queue_status",lambda x:(x!="Resolved").sum()), High_priority=("predicted_urgency",lambda x:(x=="High").sum())).reset_index()
        st.dataframe(activity, hide_index=True, use_container_width=True)
        st.markdown("### Latest inquiries"); render_tickets(df.head(5))
    else: st.info("No inquiries yet. Submit the first one from New inquiry.")

elif page == "New inquiry":
    hero("Route a new inquiry", "AI classifies urgency and sends the inquiry directly to the correct department queue.", "SMART INTAKE")
    with st.container(border=True):
        st.markdown("### Student message")
        inquiry = st.text_area("Inquiry", placeholder="Describe the issue, relevant dates, and what help is needed…", height=170, label_visibility="collapsed")
        st.info("The new BERT model predicts urgency directly from the inquiry text; no deadline input is required.")
        if st.button("Classify & route inquiry  →", type="primary", use_container_width=True):
            if not inquiry.strip(): st.warning("Enter an inquiry before submitting.")
            else:
                with st.spinner("Classifying and routing…"):
                    result = classify_inquiry(inquiry, models); triggers = find_trigger_keywords(inquiry)
                    new_id = insert_inquiry(inquiry, result["category_label"], result["urgency_label"], result["category_confidence"], result["urgency_confidence"], triggers, result["category_label"])
                st.session_state["routed"] = (new_id, result)
    if "routed" in st.session_state:
        new_id,result = st.session_state["routed"]
        st.success(f"INQ-{new_id:04d} was routed to {result['category_label']}.")
        a,b,c = st.columns(3); a.metric("Department", result["category_label"]); b.metric("Urgency", result["urgency_label"]); c.metric("Routing confidence", f"{result['category_confidence']:.0%}")

else:
    hero("Department queues", "Focused workspaces for ownership, prioritization, and resolution.", "TEAM WORKSPACES")
    department_tabs = st.tabs(departments)
    for tab, department in zip(department_tabs, departments):
        with tab:
            dept_df = df[df["department"] == department] if not df.empty else df
            c1,c2,c3 = st.columns(3)
            c1.metric("In queue", len(dept_df[dept_df["queue_status"] != "Resolved"]) if not dept_df.empty else 0)
            c2.metric("High priority", len(dept_df[(dept_df["predicted_urgency"] == "High") & (dept_df["queue_status"] != "Resolved")]) if not dept_df.empty else 0)
            c3.metric("Resolved", len(dept_df[dept_df["queue_status"] == "Resolved"]) if not dept_df.empty else 0)
            status_filter = st.segmented_control(
                "Filter", ["Open", "All", "Resolved"], default="Open", key=f"filter_{department}"
            )
            visible_df = dept_df
            if not visible_df.empty:
                if status_filter == "Open": visible_df = visible_df[visible_df["queue_status"] != "Resolved"]
                elif status_filter == "Resolved": visible_df = visible_df[visible_df["queue_status"] == "Resolved"]
            render_tickets(visible_df)
