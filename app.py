
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Campaign Conversion Efficiency Tool",
    layout="wide"
)

# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("Campaign Conversion Efficiency Tool")
st.write(
    "Upload your bank marketing dataset to identify which customer segments convert best "
    "and which ones use campaign contacts most efficiently."
)

# -----------------------------
# REQUIRED COLUMNS
# -----------------------------
REQUIRED_COLUMNS = ["age", "job", "marital", "education", "contact", "campaign", "poutcome", "y"]

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## Campaign Efficiency")
    st.caption("Bank Marketing Segment Analyzer")

    st.divider()

    st.markdown("### How to use")
    st.markdown("1. Upload a campaign CSV file")
    st.markdown("2. Use the filters to narrow the analysis")
    st.markdown("3. Review the KPI metrics and charts")
    st.markdown("4. Use the ranked table to identify efficient target segments")

    st.divider()

    st.markdown("### Required columns")
    for col in REQUIRED_COLUMNS:
        st.code(col, language=None)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

# -----------------------------
# PRE-UPLOAD MESSAGE
# -----------------------------
if uploaded_file is None:
    st.info("Upload a CSV file above to begin the analysis.")

    st.subheader("Expected data format")

    example_df = pd.DataFrame(
        {
            "age": [32, 45, 28],
            "job": ["admin.", "technician", "services"],
            "marital": ["married", "single", "married"],
            "education": ["secondary", "tertiary", "secondary"],
            "contact": ["cellular", "telephone", "cellular"],
            "campaign": [2, 4, 1],
            "poutcome": ["unknown", "failure", "success"],
            "y": ["yes", "no", "yes"],
        }
    )
    st.dataframe(example_df, use_container_width=True)
    st.stop()

# -----------------------------
# DATA LOADING AND VALIDATION
# -----------------------------
df = pd.read_csv(uploaded_file)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Please check your file.")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# DATA PREPARATION
# -----------------------------
df = df.reset_index(drop=True)
df["client_id"] = df.index + 1

filtered_df = df.copy()

# -----------------------------
# INTERACTIVE FILTERS
# -----------------------------
st.subheader("Interactive Filters")

col1, col2, col3 = st.columns(3)

with col1:
    contact_options = ["All"] + sorted(filtered_df["contact"].dropna().astype(str).unique().tolist())
    selected_contact = st.selectbox("Select Contact Method", contact_options)

with col2:
    poutcome_options = ["All"] + sorted(filtered_df["poutcome"].dropna().astype(str).unique().tolist())
    selected_poutcome = st.selectbox("Select Previous Campaign Outcome", poutcome_options)

with col3:
    marital_options = ["All"] + sorted(filtered_df["marital"].dropna().astype(str).unique().tolist())
    selected_marital = st.selectbox("Select Marital Status", marital_options)

if selected_contact != "All":
    filtered_df = filtered_df[filtered_df["contact"] == selected_contact]

if selected_poutcome != "All":
    filtered_df = filtered_df[filtered_df["poutcome"] == selected_poutcome]

if selected_marital != "All":
    filtered_df = filtered_df[filtered_df["marital"] == selected_marital]

if filtered_df.empty:
    st.warning("No records match the selected filters. Please choose different filter values.")
    st.stop()

# -----------------------------
# METRIC COMPUTATION
# -----------------------------
st.subheader("Metric Results")

filtered_df["converted"] = filtered_df["y"].astype(str).str.lower().map({"yes": 1, "no": 0})
filtered_df = filtered_df.dropna(subset=["converted"])

filtered_df["campaign"] = pd.to_numeric(filtered_df["campaign"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["campaign"])

filtered_df["age"] = pd.to_numeric(filtered_df["age"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["age"])

filtered_df["age_group"] = pd.cut(
    filtered_df["age"],
    bins=[0, 29, 44, 59, 120],
    labels=["18-29", "30-44", "45-59", "60+"],
    include_lowest=True
)

segment_results = filtered_df.groupby(["job", "age_group"], observed=False).agg(
    segment_size=("client_id", "count"),
    conversions=("converted", "sum"),
    conversion_rate=("converted", "mean"),
    average_contacts=("campaign", "mean")
).reset_index()

segment_results["average_contacts"] = segment_results["average_contacts"].fillna(0)

segment_results["efficiency_score"] = np.where(
    segment_results["average_contacts"] > 0,
    segment_results["conversion_rate"] / segment_results["average_contacts"],
    0
)

segment_results["conversion_rate"] = (segment_results["conversion_rate"] * 100).round(2)
segment_results["efficiency_score"] = segment_results["efficiency_score"].round(4)
segment_results["average_contacts"] = segment_results["average_contacts"].round(2)

segment_results = segment_results.sort_values(by="efficiency_score", ascending=False)

heatmap_results = filtered_df.groupby(["job", "education"], observed=False).agg(
    conversion_rate=("converted", "mean")
).reset_index()

heatmap_results["conversion_rate"] = (heatmap_results["conversion_rate"] * 100).round(2)
heatmap_pivot = heatmap_results.pivot(index="job", columns="education", values="conversion_rate")

overall_conversion_rate = round(filtered_df["converted"].mean() * 100, 2)

best_segment_row = segment_results.iloc[0]
best_segment_label = f"{best_segment_row['job']} | {best_segment_row['age_group']}"

total_contacts = filtered_df["campaign"].sum()
total_conversions = filtered_df["converted"].sum()

if total_conversions > 0:
    avg_contacts_per_conversion = round(total_contacts / total_conversions, 2)
else:
    avg_contacts_per_conversion = 0

# -----------------------------
# HEADLINE METRICS
# -----------------------------
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(label="Overall Conversion Rate", value=f"{overall_conversion_rate}%")

with kpi2:
    st.metric(label="Best-Performing Segment", value=best_segment_label)

with kpi3:
    st.metric(label="Average Contacts per Conversion", value=avg_contacts_per_conversion)

# -----------------------------
# RESULT TABLE
# -----------------------------
st.subheader("Ranked Segment Efficiency Table")
st.dataframe(segment_results, use_container_width=True)

# -----------------------------
# CHARTS
# -----------------------------
fig_bar = px.bar(
    segment_results,
    x="job",
    y="conversion_rate",
    color="age_group",
    barmode="group",
    title="Conversion Rate by Job Type and Age Group",
    labels={
        "job": "Job Type",
        "conversion_rate": "Conversion Rate (%)",
        "age_group": "Age Group"
    }
)
st.plotly_chart(fig_bar, use_container_width=True)

fig_scatter = px.scatter(
    segment_results,
    x="average_contacts",
    y="conversion_rate",
    size="segment_size",
    color="job",
    hover_data=["age_group", "efficiency_score"],
    title="Contact Frequency vs. Conversion Rate by Segment",
    labels={
        "average_contacts": "Average Number of Contacts",
        "conversion_rate": "Conversion Rate (%)",
        "segment_size": "Segment Size"
    }
)
st.plotly_chart(fig_scatter, use_container_width=True)

fig_heatmap = px.imshow(
    heatmap_pivot,
    text_auto=True,
    aspect="auto",
    title="Segment Efficiency Heatmap: Job × Education × Conversion Rate (%)",
    labels=dict(x="Education", y="Job", color="Conversion Rate (%)")
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# INTERPRETATION PANEL
# -----------------------------
st.info(
    f"""
    This tool shows which customer segments are responding most efficiently to the marketing campaign.
    In the current filtered view, the overall conversion rate is {overall_conversion_rate}%.
    The strongest segment right now is {best_segment_label}, meaning this group produces the best balance
    between conversion success and contact effort. A lower number of contacts combined with a stronger
    conversion rate usually indicates a more efficient segment to target. Business users should focus on
    segments with high conversion rates and lower average contacts, because those groups are producing
    better results with less campaign effort.
    """
)
