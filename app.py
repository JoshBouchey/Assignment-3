# ============================================================================
# BUSI3013 Lab 4 Tutorial — Product Velocity Index (PVI) Analyzer
# Maple & Grind Coffee Shop Analytics Tool
# ============================================================================
# This app accepts a CSV of coffee shop transaction data, computes a
# Product Velocity Index (PVI) for every menu item, classifies items
# into performance tiers, and delivers visual results a business owner
# can act on.
#
# Required CSV columns:
#   date, product_category, product_name, quantity, unit_price
# ============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Campaign Conversion Efficiency Tool", layout="wide")

st.title("Campaign Conversion Efficiency Tool")
st.write("This tool helps bank marketing teams identify which customer segments convert best and which ones use campaign contacts most efficiently.")

# --- DATA LOADING AND VALIDATION ---

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

REQUIRED_COLUMNS = ["age", "job", "marital", "education", "contact", "campaign", "poutcome", "y"]

if uploaded_file is not None:

    # STUDENT NOTE: Load the uploaded CSV file into a pandas DataFrame so the data can be cleaned, analyzed, and visualized.
    df = pd.read_csv(uploaded_file)

    # STUDENT NOTE: Validate that all required columns for Metric 13 are present before any calculations are done.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}. Please check your file.")
        st.stop()

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    # STUDENT NOTE: Create a simple client ID from the row number because this dataset does not include a unique customer ID column.
    df = df.reset_index(drop=True)
    df["client_id"] = df.index + 1

    # STUDENT NOTE: Make a copy of the uploaded data so the original DataFrame is preserved while filters are applied.
    filtered_df = df.copy()

    # --- INTERACTIVE FILTERS ---

    st.subheader("Interactive Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        # STUDENT NOTE: Allow the user to filter by contact method so the metric results change based on campaign channel.
        contact_options = ["All"] + sorted(filtered_df["contact"].dropna().astype(str).unique().tolist())
        selected_contact = st.selectbox("Select Contact Method", contact_options)

    with col2:
        # STUDENT NOTE: Allow the user to filter by previous campaign outcome so they can compare performance across customer history groups.
        poutcome_options = ["All"] + sorted(filtered_df["poutcome"].dropna().astype(str).unique().tolist())
        selected_poutcome = st.selectbox("Select Previous Campaign Outcome", poutcome_options)

    with col3:
        # STUDENT NOTE: Allow the user to filter by marital status so the output updates for different customer groups.
        marital_options = ["All"] + sorted(filtered_df["marital"].dropna().astype(str).unique().tolist())
        selected_marital = st.selectbox("Select Marital Status", marital_options)

    # STUDENT NOTE: Apply the contact method filter only when the user chooses a specific value.
    if selected_contact != "All":
        filtered_df = filtered_df[filtered_df["contact"] == selected_contact]

    # STUDENT NOTE: Apply the previous campaign outcome filter only when the user chooses a specific value.
    if selected_poutcome != "All":
        filtered_df = filtered_df[filtered_df["poutcome"] == selected_poutcome]

    # STUDENT NOTE: Apply the marital status filter only when the user chooses a specific value.
    if selected_marital != "All":
        filtered_df = filtered_df[filtered_df["marital"] == selected_marital]

    # STUDENT NOTE: Stop the app safely if the filters remove all rows, so charts and metrics do not break.
    if filtered_df.empty:
        st.warning("No records match the selected filters. Please choose different filter values.")
        st.stop()

    # --- METRIC COMPUTATION ---

    st.subheader("Metric Results")

    # STUDENT NOTE: Convert the subscription result into numeric form so conversion rates can be calculated as averages.
    filtered_df["converted"] = filtered_df["y"].map({"yes": 1, "no": 0})

    # STUDENT NOTE: Remove rows where conversion could not be mapped properly so the metric calculations stay accurate.
    filtered_df = filtered_df.dropna(subset=["converted"])

    # STUDENT NOTE: Convert the campaign column to numeric format in case it was read as text from the uploaded file.
    filtered_df["campaign"] = pd.to_numeric(filtered_df["campaign"], errors="coerce")

    # STUDENT NOTE: Drop rows with missing campaign values because the efficiency score depends on the number of contacts.
    filtered_df = filtered_df.dropna(subset=["campaign"])

    # STUDENT NOTE: Create age groups so the output is easier for business users to interpret than raw ages.
    filtered_df["age_group"] = pd.cut(
        filtered_df["age"],
        bins=[0, 29, 44, 59, 120],
        labels=["18-29", "30-44", "45-59", "60+"],
        include_lowest=True
    )

    # STUDENT NOTE: Group the data by job and age group to calculate conversion rate and average contacts for each customer segment.
    segment_results = filtered_df.groupby(["job", "age_group"], observed=False).agg(
        segment_size=("client_id", "count"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"),
        average_contacts=("campaign", "mean")
    ).reset_index()

    # STUDENT NOTE: Replace missing average contact values with zero before calculating the efficiency score.
    segment_results["average_contacts"] = segment_results["average_contacts"].fillna(0)

    # STUDENT NOTE: Calculate the conversion efficiency score as conversion rate divided by average contacts so segments that convert well with fewer contacts rank higher.
    segment_results["efficiency_score"] = np.where(
        segment_results["average_contacts"] > 0,
        segment_results["conversion_rate"] / segment_results["average_contacts"],
        0
    )

    # STUDENT NOTE: Convert conversion rate and efficiency score into rounded values for clearer business reporting.
    segment_results["conversion_rate"] = (segment_results["conversion_rate"] * 100).round(2)
    segment_results["efficiency_score"] = segment_results["efficiency_score"].round(4)
    segment_results["average_contacts"] = segment_results["average_contacts"].round(2)

    # STUDENT NOTE: Sort segments from highest to lowest efficiency so the best-performing groups appear first.
    segment_results = segment_results.sort_values(by="efficiency_score", ascending=False)

    # STUDENT NOTE: Build a second grouped table by job and education for the heatmap required by the assignment.
    heatmap_results = filtered_df.groupby(["job", "education"], observed=False).agg(
        conversion_rate=("converted", "mean")
    ).reset_index()

    # STUDENT NOTE: Convert the heatmap conversion rates to percentages for easier interpretation.
    heatmap_results["conversion_rate"] = (heatmap_results["conversion_rate"] * 100).round(2)

    # STUDENT NOTE: Create a pivot table so job is shown on one axis, education on the other, and conversion rate in the cells.
    heatmap_pivot = heatmap_results.pivot(index="job", columns="education", values="conversion_rate")

    # STUDENT NOTE: Calculate the overall conversion rate for the filtered dataset as a headline KPI.
    overall_conversion_rate = round(filtered_df["converted"].mean() * 100, 2)

    # STUDENT NOTE: Identify the best-performing segment based on the highest efficiency score.
    best_segment_row = segment_results.iloc[0]
    best_segment_label = f"{best_segment_row['job']} | {best_segment_row['age_group']}"

    # STUDENT NOTE: Calculate average contacts per conversion by dividing total campaign contacts by total successful conversions.
    total_contacts = filtered_df["campaign"].sum()
    total_conversions = filtered_df["converted"].sum()

    # STUDENT NOTE: Avoid division by zero if no conversions are found in the filtered dataset.
    if total_conversions > 0:
        avg_contacts_per_conversion = round(total_contacts / total_conversions, 2)
    else:
        avg_contacts_per_conversion = 0

    # --- HEADLINE METRICS ---

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(label="Overall Conversion Rate", value=f"{overall_conversion_rate}%")

    with kpi2:
        st.metric(label="Best-Performing Segment", value=best_segment_label)

    with kpi3:
        st.metric(label="Average Contacts per Conversion", value=avg_contacts_per_conversion)

    # --- RESULT TABLE ---

    st.subheader("Ranked Segment Efficiency Table")
    st.dataframe(segment_results)

    # --- CHARTS ---

    # STUDENT NOTE: Create a bar chart showing conversion rate by job type and age group so users can compare segment performance visually.
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

    # STUDENT NOTE: Create a scatter plot showing average contact frequency versus conversion rate to examine efficiency patterns across segments.
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

    # STUDENT NOTE: Create a heatmap showing conversion percentages across job and education combinations to highlight strong and weak customer segments.
    fig_heatmap = px.imshow(
        heatmap_pivot,
        text_auto=True,
        aspect="auto",
        title="Segment Efficiency Heatmap: Job × Education × Conversion Rate (%)",
        labels=dict(x="Education", y="Job", color="Conversion Rate (%)")
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- INTERPRETATION ---

    # STUDENT NOTE: Provide a plain-English explanation so non-technical users understand what the current results mean and what to focus on.
    st.info(
        f"""
        This tool shows which customer segments are responding most efficiently to the marketing campaign.
        In the current filtered view, the overall conversion rate is {overall_conversion_rate}%.
        The strongest segment right now is {best_segment_label}, meaning this group produces the best balance
        between conversion success and contact effort. A lower number of contacts combined with a stronger
        conversion rate usually indicates a more efficient segment to target. Business users should pay close
        attention to segments with high conversion rates but lower average contacts, because those groups are
        giving better results with less campaign effort.
        """
    )
