import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.pipeline import run_pipeline
from app.config import RAW_DATA_DIR, CLEANED_DATA_FILE, REPORT_DIR, CHART_DIR


def load_preview(file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin1")
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")


def load_cleaned_data():
    if not Path(CLEANED_DATA_FILE).exists():
        return None

    try:
        return pd.read_csv(CLEANED_DATA_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(CLEANED_DATA_FILE, encoding="latin1")


def clear_display_outputs():
    for folder in [CHART_DIR, REPORT_DIR]:
        folder_path = Path(folder)
        if folder_path.exists():
            for file in folder_path.iterdir():
                if file.is_file():
                    file.unlink()

    if Path(CLEANED_DATA_FILE).exists():
        Path(CLEANED_DATA_FILE).unlink()


def format_number(value):
    try:
        return f"{value:,.2f}"
    except Exception:
        return str(value)


def answer_dataset_question(question: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No analyzed dataset is available yet. Upload a dataset and run analysis first."

    q = question.lower().strip()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if "rows" in q or "records" in q or "how many rows" in q:
        return f"The dataset contains {len(df):,} rows."

    if "columns" in q or "features" in q:
        return f"The dataset has {df.shape[1]} columns: {', '.join(df.columns.tolist())}"

    if "missing" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return "There are no missing values in the cleaned dataset."
        return "Missing values by column:\n" + "\n".join(
            [f"- {col}: {int(val)}" for col, val in missing.items()]
        )

    if "anomaly" in q or "outlier" in q:
        if "anomaly_flag" in df.columns:
            anomaly_count = int(df["anomaly_flag"].sum())
            pct = (anomaly_count / len(df)) * 100 if len(df) else 0
            return f"Detected {anomaly_count} anomalies, which is {pct:.2f}% of the dataset."
        return "This dataset does not currently include an anomaly flag."

    if "sales" in q and "total" in q and "Sales" in df.columns:
        return f"Total Sales = {format_number(df['Sales'].sum())}"

    if "profit" in q and "total" in q and "Profit" in df.columns:
        return f"Total Profit = {format_number(df['Profit'].sum())}"

    if "average" in q or "mean" in q:
        for col in numeric_cols:
            if col.lower() in q:
                return f"Average {col} = {format_number(df[col].mean())}"
        if numeric_cols:
            col = numeric_cols[0]
            return f"Average {col} = {format_number(df[col].mean())}"

    if "max" in q or "highest" in q or "top" in q:
        for col in numeric_cols:
            if col.lower() in q:
                return f"Highest {col} = {format_number(df[col].max())}"
        if "category" in q and "Category" in df.columns:
            top_category = df["Category"].value_counts().idxmax()
            return f"The most frequent category is {top_category}."

    if "min" in q or "lowest" in q:
        for col in numeric_cols:
            if col.lower() in q:
                return f"Lowest {col} = {format_number(df[col].min())}"

    if "top category" in q and "Category" in df.columns:
        if "Sales" in df.columns:
            grouped = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            return f"Top category by Sales is {grouped.index[0]} with {format_number(grouped.iloc[0])}."
        return f"Most frequent category is {df['Category'].value_counts().idxmax()}."

    if "top region" in q and "Region" in df.columns:
        if "Sales" in df.columns:
            grouped = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
            return f"Top region by Sales is {grouped.index[0]} with {format_number(grouped.iloc[0])}."
        return f"Most frequent region is {df['Region'].value_counts().idxmax()}."

    if "correlation" in q:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    pairs.append(
                        (corr.columns[i], corr.columns[j], abs(corr.iloc[i, j]), corr.iloc[i, j])
                    )
            pairs.sort(key=lambda x: x[2], reverse=True)
            if pairs:
                a, b, _, val = pairs[0]
                return f"The strongest numeric correlation is between {a} and {b}: {val:.2f}"
        return "Not enough numeric columns to compute correlation."

    if "summary" in q or "describe" in q:
        lines = [
            f"Rows: {len(df):,}",
            f"Columns: {df.shape[1]}",
            f"Numeric columns: {len(numeric_cols)}",
            f"Categorical columns: {len(categorical_cols)}",
        ]
        return "\n".join(lines)

    return (
        "I can answer questions about rows, columns, missing values, anomalies, totals, averages, top categories, "
        "top regions, and correlations. Try something like:\n"
        "- total sales\n"
        "- total profit\n"
        "- how many rows\n"
        "- missing values\n"
        "- anomaly count\n"
        "- top category by sales"
    )


st.set_page_config(
    page_title="Autonomous Data Analyst Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #111827, #172033);
        color: white;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: 0.3px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 32px;
    }

    .login-shell {
        max-width: 520px;
        margin: 50px auto 0 auto;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 36px;
        box-shadow: 0 18px 50px rgba(0,0,0,0.30);
        backdrop-filter: blur(10px);
    }

    .panel {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(37,99,235,0.20), rgba(255,255,255,0.04));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        min-height: 110px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .metric-label {
        font-size: 14px;
        color: #cbd5e1;
        margin-bottom: 10px;
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
    }

    .top-banner {
        background: linear-gradient(90deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }

    .section-heading {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .muted-text {
        color: #94a3b8;
        font-size: 14px;
    }

    .status-good {
        color: #22c55e;
        font-weight: 700;
    }

    .status-wait {
        color: #fbbf24;
        font-weight: 700;
    }

    div[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 700;
        padding: 0.7rem 1rem;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        color: white;
    }

    .report-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .small-note {
        text-align: center;
        font-size: 13px;
        color: #94a3b8;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hi. Upload a dataset, run analysis, and then ask me questions about the analyzed data."}
    ]


def show_login_page():
    st.markdown('<div class="hero-title">Autonomous Data Analyst Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Professional AI-powered analytics platform for cleaning, exploration, modeling, anomaly detection, and insight generation.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    st.markdown("## Welcome back")
    st.markdown('<div class="muted-text">Sign in to access your analytics workspace</div>', unsafe_allow_html=True)

    username = st.text_input("Email or Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Sign In"):
        if username.strip() and password.strip():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Please enter username and password.")

    st.markdown(
        '<div class="small-note">Demo login for project showcase. Any non-empty username and password will work.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def show_chatbot():
    st.subheader("Dataset Chat Assistant")

    df = load_cleaned_data()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something about the analyzed dataset...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = answer_dataset_question(prompt, df)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.markdown('</div>', unsafe_allow_html=True)


def show_dashboard():
    st.sidebar.title("Workspace")
    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.analysis_done = False
        st.session_state.last_uploaded_file = None
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hi. Upload a dataset, run analysis, and then ask me questions about the analyzed data."}
        ]
        st.rerun()

    st.markdown(
        """
        <div class="top-banner">
            <div class="section-heading">Analytics Workspace</div>
            <div class="muted-text">
                Upload a dataset, select the target column, and generate a fresh end-to-end analysis with reports, insights, anomaly detection, and visualizations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Dataset Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        save_path = Path(RAW_DATA_DIR) / uploaded_file.name

        if st.session_state.last_uploaded_file != uploaded_file.name:
            clear_display_outputs()
            st.session_state.analysis_done = False
            st.session_state.last_uploaded_file = uploaded_file.name
            st.session_state.chat_messages = [
                {"role": "assistant", "content": f"Dataset `{uploaded_file.name}` uploaded. Run analysis, then ask me questions about it."}
            ]

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df_preview = load_preview(save_path)

        excluded_keywords = ["id", "date", "name", "order id", "customer id", "product id", "postal code"]
        suggested_columns = [
            col for col in df_preview.columns
            if not any(word in col.lower() for word in excluded_keywords)
        ]
        target_options = suggested_columns if suggested_columns else df_preview.columns.tolist()

        target_column = st.sidebar.selectbox(
            "Select Target Column for ML",
            options=target_options
        )

        if st.sidebar.button("Run Full Analysis"):
            with st.spinner("Running autonomous analytics pipeline..."):
                run_pipeline(file_path=save_path, target_column=target_column)

            st.session_state.analysis_done = True
            st.success("Analysis completed successfully.")

        left, right = st.columns([2, 1])

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Dataset Preview")
            st.dataframe(df_preview.head(20), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("Dataset Summary")
            st.write(f"**File Name:** {uploaded_file.name}")
            st.write(f"**Rows:** {df_preview.shape[0]}")
            st.write(f"**Columns:** {df_preview.shape[1]}")
            st.write(f"**Selected Target:** {target_column}")
            if st.session_state.analysis_done:
                st.markdown('<span class="status-good">Status: Analysis Complete</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-wait">Status: Waiting for Analysis</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.analysis_done and Path(CLEANED_DATA_FILE).exists():
            df = load_cleaned_data()

            st.subheader("Key Metrics")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Total Records</div>
                        <div class="metric-number">{len(df):,}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                sales_value = f"{df['Sales'].sum():,.2f}" if "Sales" in df.columns else "N/A"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Total Sales</div>
                        <div class="metric-number">{sales_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:
                profit_value = f"{df['Profit'].sum():,.2f}" if "Profit" in df.columns else "N/A"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Total Profit</div>
                        <div class="metric-number">{profit_value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c4:
                anomaly_count = int(df["anomaly_flag"].sum()) if "anomaly_flag" in df.columns else 0
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-label">Anomalies</div>
                        <div class="metric-number">{anomaly_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            menu = st.sidebar.radio(
                "Navigation",
                ["Dashboard", "Reports", "Insights", "Visualizations", "Chat Assistant"]
            )

            if menu == "Dashboard":
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.subheader("Cleaned Dataset")
                st.dataframe(df.head(50), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            elif menu == "Reports":
                st.subheader("Generated Reports")
                cleaning_report = REPORT_DIR / "cleaning_report.txt"
                model_report = REPORT_DIR / "model_report.txt"

                if cleaning_report.exists():
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.markdown("### Cleaning Report")
                    st.download_button(
                        "Download Cleaning Report",
                        data=cleaning_report.read_text(encoding="utf-8"),
                        file_name="cleaning_report.txt"
                    )
                    st.text(cleaning_report.read_text(encoding="utf-8"))
                    st.markdown('</div>', unsafe_allow_html=True)

                if model_report.exists():
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.markdown("### Model Report")
                    st.download_button(
                        "Download Model Report",
                        data=model_report.read_text(encoding="utf-8"),
                        file_name="model_report.txt"
                    )
                    st.text(model_report.read_text(encoding="utf-8"))
                    st.markdown('</div>', unsafe_allow_html=True)

            elif menu == "Insights":
                st.subheader("AI Insights")
                insight_report = REPORT_DIR / "insights.txt"

                if insight_report.exists():
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.download_button(
                        "Download Insights",
                        data=insight_report.read_text(encoding="utf-8"),
                        file_name="insights.txt"
                    )
                    st.text(insight_report.read_text(encoding="utf-8"))
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Run analysis to generate insights.")

            elif menu == "Visualizations":
                st.subheader("Visualization Gallery")
                chart_files = sorted(Path(CHART_DIR).glob("*.png"))

                if chart_files:
                    for i in range(0, len(chart_files), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(chart_files):
                                with cols[j]:
                                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                                    st.image(
                                        str(chart_files[i + j]),
                                        caption=chart_files[i + j].name,
                                        use_container_width=True
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No charts available.")

            elif menu == "Chat Assistant":
                show_chatbot()
        else:
            st.info("Upload a dataset and click 'Run Full Analysis' to generate fresh outputs.")
    else:
        st.info("Upload a dataset from the sidebar to begin.")


if not st.session_state.logged_in:
    show_login_page()
else:
    show_dashboard()