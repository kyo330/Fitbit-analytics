import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# Page config
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Styles
# Text is consistently light so it reads on the dark background.
# Key rules: labels >= #c2cadb, body text >= #e4eafa, headings #f1f5fb.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.page-title {
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c9ef5, #a78bfa, #60d9fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #aeb6cc;
    font-size: 0.88rem;
    margin-bottom: 24px;
}

.section-heading {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5fb;
    border-left: 4px solid #7c9ef5;
    padding-left: 12px;
    margin: 28px 0 16px;
}

.kpi-card {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #353b58;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #c2cadb;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}

.kpi-sub {
    font-size: 0.8rem;
    color: #aeb6cc;
    margin-top: 4px;
}

.insight-card {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.92rem;
    color: #e4eafa;
    line-height: 1.65;
}

.insight-card b {
    color: #ffffff;
}

.insight-good {
    background: linear-gradient(135deg, #112418, #13291d);
    border-left: 4px solid #48bb78;
}

.insight-warn {
    background: linear-gradient(135deg, #241d10, #2c2413);
    border-left: 4px solid #f6a623;
}

.insight-neutral {
    background: linear-gradient(135deg, #1c2238, #222a48);
    border-left: 4px solid #7c9ef5;
}

.persona-card {
    background: linear-gradient(135deg, #1a2040, #1e2850);
    border: 1px solid #3a4a7a;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.persona-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}

.persona-detail {
    color: #c2cadb;
    font-size: 0.85rem;
    line-height: 1.7;
}

h1, h2, h3 { color: #f1f5fb !important; }
.stSidebar { background: #141823 !important; }
section[data-testid="stSidebar"] * { color: #dce3f2 !important; }
div[data-testid="stMetricValue"] { color: #7c9ef5; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# Color palette used across all charts
PALETTE = ["#60d9fa", "#a78bfa", "#48bb78", "#f6a623", "#fc8181", "#f687b3", "#7c9ef5"]

# Shared Plotly layout settings applied to every chart
CHART_DEFAULTS = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#cbd5e8",
    font_family="Inter",
    title_font_color="#f1f5fb",
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#cbd5e8"),
    xaxis=dict(gridcolor="#272d42", showgrid=True, zeroline=False, color="#cbd5e8"),
    yaxis=dict(gridcolor="#272d42", showgrid=True, zeroline=False, color="#cbd5e8"),
    margin=dict(l=10, r=10, t=48, b=10),
    height=360,
)


# Data loading

@st.cache_data
def load_data():
    
    df = pd.read_csv(Customers.csv)

    # Normalize column names to a consistent set
    column_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("customerid", "id", "customer id"):
            column_map[col] = "CustomerID"
        elif lower in ("gender", "genre", "sex"):
            column_map[col] = "Gender"
        elif lower == "age":
            column_map[col] = "Age"
        elif "income" in lower:
            column_map[col] = "AnnualIncome"
        elif "spending" in lower:
            column_map[col] = "SpendingScore"
        elif "profession" in lower or "occupation" in lower:
            column_map[col] = "Profession"
        elif "work" in lower and "exp" in lower:
            column_map[col] = "WorkExperience"
        elif "family" in lower:
            column_map[col] = "FamilySize"

    df = df.rename(columns=column_map)

    # Basic cleaning
    if "Age" in df.columns:
        df = df[df["Age"] > 0]
    if "AnnualIncome" in df.columns:
        df = df[df["AnnualIncome"] >= 0]
    if "Profession" in df.columns:
        df["Profession"] = df["Profession"].fillna("Unknown").replace("", "Unknown")
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.strip().str.title()

    return df.reset_index(drop=True)


# Segmentation

@st.cache_data
def segment_customers(df, k=5):
    """
    KMeans clustering on annual income, spending score, and age.
    Each cluster gets a readable persona name based on its income and spending profile.
    Returns the dataframe with Segment and SegmentName columns added.
    """
    feature_cols = [c for c in ["AnnualIncome", "SpendingScore", "Age"] if c in df.columns]
    clean = df.dropna(subset=feature_cols).copy()

    scaled = StandardScaler().fit_transform(clean[feature_cols])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clean["Segment"] = km.fit_predict(scaled)

    income_mid = clean["AnnualIncome"].median()
    spending_mid = clean["SpendingScore"].median()

    def label_segment(group_df):
        high_income = group_df["AnnualIncome"].mean() >= income_mid
        high_spending = group_df["SpendingScore"].mean() >= spending_mid
        if high_income and high_spending:
            return "Premium Loyalists"
        elif high_income and not high_spending:
            return "Untapped High-Earners"
        elif not high_income and high_spending:
            return "Aspirational Spenders"
        else:
            return "Budget Conscious"

    raw_labels = {seg: label_segment(grp) for seg, grp in clean.groupby("Segment")}

    # If two segments get the same name, distinguish by age
    seen = {}
    final_labels = {}
    for seg in sorted(clean["Segment"].unique()):
        name = raw_labels[seg]
        if name in seen:
            avg_age = clean.loc[clean["Segment"] == seg, "Age"].mean()
            suffix = "Younger" if avg_age < clean["Age"].median() else "Older"
            final_labels[seg] = f"{name} ({suffix})"
        else:
            seen[name] = True
            final_labels[seg] = name

    clean["SegmentName"] = clean["Segment"].map(final_labels)
    return clean, final_labels


# Shared UI components

def kpi_card(value, label, sub="", color="#7c9ef5"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def apply_chart_style(fig, title="", height=360):
    settings = {**CHART_DEFAULTS, "height": height}
    if title:
        settings["title"] = title
    fig.update_layout(**settings)
    return fig


# Overview page

def show_overview(df):
    st.markdown('<div class="page-title">Customer Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Kaggle: datascientistanna/customers-dataset &nbsp;|&nbsp; '
        f'{len(df):,} customers &nbsp;|&nbsp; demographics, income, and spending behavior</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    with cols[0]:
        kpi_card(f"{len(df):,}", "Total Customers", "in dataset", "#60d9fa")
    with cols[1]:
        kpi_card(
            f"${df['AnnualIncome'].mean():,.0f}",
            "Avg Annual Income",
            f"median ${df['AnnualIncome'].median():,.0f}",
            "#48bb78",
        )
    with cols[2]:
        kpi_card(
            f"{df['SpendingScore'].mean():.0f} / 100",
            "Avg Spending Score",
            f"std {df['SpendingScore'].std():.0f}",
            "#a78bfa",
        )
    with cols[3]:
        kpi_card(
            f"{df['Age'].mean():.0f} yrs",
            "Avg Age",
            f"range {int(df['Age'].min())} to {int(df['Age'].max())}",
            "#f6a623",
        )

    st.markdown("")

    row2 = st.columns(4)
    if "Gender" in df.columns:
        female_pct = (df["Gender"] == "Female").mean() * 100
        with row2[0]:
            kpi_card(f"{female_pct:.0f}% F / {100 - female_pct:.0f}% M", "Gender Split", "", "#f687b3")
    if "Profession" in df.columns:
        with row2[1]:
            kpi_card(f"{df['Profession'].nunique()}", "Professions", "distinct categories", "#7c9ef5")
    if "FamilySize" in df.columns:
        with row2[2]:
            kpi_card(f"{df['FamilySize'].mean():.1f}", "Avg Family Size", "members per household", "#60d9fa")
    if "WorkExperience" in df.columns:
        with row2[3]:
            kpi_card(f"{df['WorkExperience'].mean():.1f} yrs", "Avg Work Experience", "", "#48bb78")

    section("Data Preview")
    preview_cols = [c for c in
        ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore",
         "Profession", "WorkExperience", "FamilySize"]
        if c in df.columns]
    st.dataframe(df[preview_cols].head(25), use_container_width=True, height=340)

    section("Summary Statistics")
    num_cols = [c for c in ["Age", "AnnualIncome", "SpendingScore", "WorkExperience", "FamilySize"]
                if c in df.columns]
    st.dataframe(df[num_cols].describe().round(1), use_container_width=True)


# Demographics page

def show_demographics(df):
    section("Age and Income Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age", nbins=25, color_discrete_sequence=["#60d9fa"])
        apply_chart_style(fig, "How old are our customers?", 320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="AnnualIncome", nbins=25, color_discrete_sequence=["#48bb78"],
                           labels={"AnnualIncome": "Annual Income ($)"})
        apply_chart_style(fig, "How much do they earn?", 320)
        st.plotly_chart(fig, use_container_width=True)

    section("Gender and Profession Breakdown")

    col3, col4 = st.columns(2)
    if "Gender" in df.columns:
        with col3:
            counts = df["Gender"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.45,
                marker_colors=["#f687b3", "#60d9fa"],
                textinfo="label+percent",
                textfont_size=13,
            ))
            apply_chart_style(fig, "Gender Split", 320)
            st.plotly_chart(fig, use_container_width=True)

    if "Profession" in df.columns:
        with col4:
            top = df["Profession"].value_counts().head(10).sort_values()
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color="#a78bfa",
                text=top.values, textposition="outside",
                textfont_color="#e4eafa",
            ))
            apply_chart_style(fig, "Top 10 Professions", 320)
            st.plotly_chart(fig, use_container_width=True)

    section("Who Spends the Most?")

    col5, col6 = st.columns(2)

    with col5:
        df["AgeBand"] = pd.cut(
            df["Age"], bins=[0, 25, 35, 45, 55, 100],
            labels=["Under 25", "25 to 34", "35 to 44", "45 to 54", "55 plus"]
        )
        age_spend = df.groupby("AgeBand", observed=True)["SpendingScore"].mean().round(1)
        fig = go.Figure(go.Bar(
            x=age_spend.index.astype(str),
            y=age_spend.values,
            marker_color=PALETTE[:len(age_spend)],
            text=age_spend.values, textposition="outside",
            textfont_color="#e4eafa",
        ))
        apply_chart_style(fig, "Avg Spending Score by Age Group", 320)
        st.plotly_chart(fig, use_container_width=True)

    if "Gender" in df.columns:
        with col6:
            gender_spend = df.groupby("Gender")["SpendingScore"].mean().round(1)
            fig = go.Figure(go.Bar(
                x=gender_spend.index,
                y=gender_spend.values,
                marker_color=["#f687b3", "#60d9fa"],
                text=gender_spend.values, textposition="outside",
                textfont_color="#e4eafa",
            ))
            apply_chart_style(fig, "Avg Spending Score by Gender", 320)
            st.plotly_chart(fig, use_container_width=True)

    if "Profession" in df.columns:
        section("Spending by Profession")
        # Only include professions with enough customers to be meaningful
        prof_size = df["Profession"].value_counts()
        valid_profs = prof_size[prof_size >= 15].index
        prof_spend = (df[df["Profession"].isin(valid_profs)]
                      .groupby("Profession")["SpendingScore"]
                      .mean()
                      .round(1)
                      .sort_values(ascending=True))

        fig = go.Figure(go.Bar(
            x=prof_spend.values,
            y=prof_spend.index,
            orientation="h",
            marker_color="#7c9ef5",
            text=prof_spend.values, textposition="outside",
            textfont_color="#e4eafa",
        ))
        apply_chart_style(fig, "Avg Spending Score by Profession (min 15 customers)", 420)
        st.plotly_chart(fig, use_container_width=True)


# Segmentation page

def show_segmentation(df):
    section("KMeans Customer Segmentation")

    k = st.slider("Number of segments", min_value=3, max_value=7, value=5,
                  help="Adjust this to see how the clusters change.")

    seg_df, label_map = segment_customers(df, k)

    # Main scatter plot: income vs spending colored by segment
    fig = px.scatter(
        seg_df,
        x="AnnualIncome", y="SpendingScore",
        color="SegmentName",
        size="Age",
        color_discrete_sequence=PALETTE,
        labels={
            "AnnualIncome": "Annual Income ($)",
            "SpendingScore": "Spending Score (1 to 100)",
            "SegmentName": "Segment",
        },
        title="Customer Segments: Income vs Spending (bubble size reflects age)",
        opacity=0.75,
    )
    apply_chart_style(fig, height=480)
    st.plotly_chart(fig, use_container_width=True)

    section("Segment Personas")

    summary = (
        seg_df.groupby("SegmentName")
        .agg(
            customers=("CustomerID", "count"),
            avg_income=("AnnualIncome", "mean"),
            avg_spending=("SpendingScore", "mean"),
            avg_age=("Age", "mean"),
        )
        .round(0)
        .sort_values("avg_spending", ascending=False)
    )

    total = len(seg_df)
    persona_cols = st.columns(2)
    for i, (name, row) in enumerate(summary.iterrows()):
        share = row["customers"] / total * 100
        with persona_cols[i % 2]:
            st.markdown(f"""
            <div class="persona-card">
                <div class="persona-name">{name}</div>
                <div class="persona-detail">
                    {int(row['customers'])} customers ({share:.0f}% of base)<br>
                    Avg income: ${row['avg_income']:,.0f}
                    &nbsp; Avg spending: {row['avg_spending']:.0f} / 100
                    &nbsp; Avg age: {row['avg_age']:.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    section("Segment Comparison Table")

    display = summary.copy()
    display["avg_income"] = display["avg_income"].apply(lambda x: f"${x:,.0f}")
    display["avg_spending"] = display["avg_spending"].apply(lambda x: f"{x:.0f} / 100")
    display["avg_age"] = display["avg_age"].apply(lambda x: f"{x:.0f} yrs")
    display.columns = ["Customers", "Avg Income", "Avg Spending Score", "Avg Age"]
    st.dataframe(display, use_container_width=True)

    section("Segment Distribution")
    dist = seg_df["SegmentName"].value_counts()
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values,
        marker_color=PALETTE[:len(dist)],
        text=dist.values, textposition="outside",
        textfont_color="#e4eafa",
    ))
    apply_chart_style(fig, "How many customers are in each segment?", 340)
    st.plotly_chart(fig, use_container_width=True)


# Relationships page

def show_relationships(df):
    section("Correlation Matrix")

    num_cols = [c for c in ["Age", "AnnualIncome", "SpendingScore", "WorkExperience", "FamilySize"]
                if c in df.columns]
    corr = df[num_cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=num_cols,
        y=num_cols,
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=12, color="#ffffff"),
        colorbar=dict(thickness=14, tickfont_color="#cbd5e8"),
    ))
    apply_chart_style(fig, "Pearson Correlation Between Key Variables", 460)
    st.plotly_chart(fig, use_container_width=True)

    section("Key Pairwise Relationships")
    st.markdown('<p style="color:#aeb6cc;font-size:0.88rem;margin-top:-12px;margin-bottom:16px;">'
                'Each chart shows the trend line and Pearson r value.</p>', unsafe_allow_html=True)

    pairs = [
        ("AnnualIncome", "SpendingScore", "#f687b3", "Does earning more mean spending more?"),
        ("Age", "SpendingScore", "#48bb78", "Do older customers spend less?"),
        ("Age", "AnnualIncome", "#60d9fa", "Does income rise with age?"),
    ]

    scatter_cols = st.columns(3)
    for col, (x, y, color, question) in zip(scatter_cols, pairs):
        valid = df[[x, y]].dropna()
        r, p = stats.pearsonr(valid[x], valid[y])
        fig = px.scatter(
            df, x=x, y=y,
            trendline="ols",
            opacity=0.4,
            color_discrete_sequence=[color],
            labels={x: x, y: y},
        )
        apply_chart_style(fig, f"{question}<br><sup>r = {r:.3f}, p = {p:.1e}</sup>", 300)
        with col:
            st.plotly_chart(fig, use_container_width=True)

    if "FamilySize" in df.columns:
        section("Family Size and Spending")
        family_spend = df.groupby("FamilySize")["SpendingScore"].mean().round(1)
        fig = go.Figure(go.Scatter(
            x=family_spend.index.astype(str),
            y=family_spend.values,
            mode="lines+markers",
            line=dict(color="#a78bfa", width=2.5),
            marker=dict(size=9, color="#a78bfa"),
        ))
        apply_chart_style(fig, "Does family size affect spending behavior?", 320)
        st.plotly_chart(fig, use_container_width=True)


# Insights page

def show_insights(df):
    section("Marketing Takeaways")
    st.markdown(
        '<p style="color:#aeb6cc;font-size:0.88rem;margin-top:-12px;margin-bottom:20px;">'
        "Plain-language findings from the data, framed for a marketing team.</p>",
        unsafe_allow_html=True,
    )

    seg_df, _ = segment_customers(df, 5)
    seg_summary = (
        seg_df.groupby("SegmentName")
        .agg(n=("CustomerID", "count"), spend=("SpendingScore", "mean"), income=("AnnualIncome", "mean"))
    )

    insights = []

    # Best segment to target
    top_seg = seg_summary.sort_values("spend", ascending=False).index[0]
    top = seg_summary.loc[top_seg]
    insights.append(("good",
        f"The <b>{top_seg}</b> segment has the highest average spending score "
        f"({top['spend']:.0f} out of 100) and represents "
        f"{top['n'] / len(seg_df) * 100:.0f}% of the customer base. "
        f"This is the best group to prioritize for retention and upsell campaigns."))

    # Untapped earners
    untapped = [s for s in seg_summary.index if "Untapped" in s]
    if untapped:
        ut = seg_summary.loc[untapped[0]]
        insights.append(("warn",
            f"The <b>{untapped[0]}</b> group earns well (avg ${ut['income']:,.0f}) "
            f"but scores only {ut['spend']:.0f} on spending. "
            f"There is real conversion opportunity here worth investigating with targeted messaging."))

    # Income vs spending correlation
    r_inc_spend, _ = stats.pearsonr(df["AnnualIncome"], df["SpendingScore"])
    if abs(r_inc_spend) < 0.15:
        insights.append(("neutral",
            f"Income and spending score are barely correlated (r = {r_inc_spend:.3f}). "
            f"This is a useful finding: income alone is a weak targeting signal for this audience. "
            f"Behavioral and demographic variables will do more work."))
    else:
        direction = "positive" if r_inc_spend > 0 else "negative"
        insights.append(("good" if r_inc_spend > 0 else "warn",
            f"There is a {direction} relationship between income and spending score "
            f"(r = {r_inc_spend:.3f}). "
            f"Income can be used as a targeting variable for this audience."))

    # Age and spending
    young_spend = df[df["Age"] < 35]["SpendingScore"].mean()
    older_spend = df[df["Age"] >= 35]["SpendingScore"].mean()
    younger_higher = young_spend > older_spend
    insights.append(("good" if younger_higher else "neutral",
        f"Customers under 35 score {young_spend:.0f} on spending vs {older_spend:.0f} for those 35 and over. "
        f"{'Younger customers are the more active spenders, worth prioritizing in acquisition campaigns.' if younger_higher else 'Older customers spend at a comparable rate, so age-based targeting needs nuance.'}"))

    # Gender spending gap
    if "Gender" in df.columns:
        gs = df.groupby("Gender")["SpendingScore"].mean()
        if len(gs) == 2:
            high_g, low_g = gs.idxmax(), gs.idxmin()
            gap = gs.max() - gs.min()
            insights.append(("neutral",
                f"{high_g} customers average {gs.max():.0f} on the spending score vs "
                f"{gs.min():.0f} for {low_g} customers, a gap of {gap:.1f} points. "
                f"{'This is a meaningful difference worth testing gender-specific creative against.' if gap > 5 else 'The gap is small, so gender alone is unlikely to be a strong segmentation lever.'}"))

    # Top spending profession
    if "Profession" in df.columns:
        prof_size = df["Profession"].value_counts()
        valid = prof_size[prof_size >= 15].index
        if len(valid):
            prof_spend = df[df["Profession"].isin(valid)].groupby("Profession")["SpendingScore"].mean()
            top_prof = prof_spend.idxmax()
            insights.append(("good",
                f"Among well-represented professions, <b>{top_prof}</b> customers score highest on spending "
                f"({prof_spend.max():.0f} out of 100). "
                f"This group is worth testing targeted creative against."))

    # Family size note
    if "FamilySize" in df.columns:
        r_fam, _ = stats.pearsonr(
            df["FamilySize"].dropna(),
            df.loc[df["FamilySize"].notna(), "SpendingScore"]
        )
        insights.append(("neutral",
            f"Family size has a {'weak' if abs(r_fam) < 0.2 else 'moderate'} relationship with spending "
            f"(r = {r_fam:.3f}). "
            f"{'It is not a strong standalone targeting variable but could add value in a combined model.' if abs(r_fam) < 0.2 else 'It may be worth including in a combined segmentation model.'}"))

    style_map = {"good": "insight-good", "warn": "insight-warn", "neutral": "insight-neutral"}
    for tone, text in insights:
        css_class = style_map.get(tone, "insight-neutral")
        st.markdown(f'<div class="insight-card {css_class}">{text}</div>', unsafe_allow_html=True)

    # Population summary chart
    section("Population Summary")
    pop_metrics = {}
    pop_metrics["Avg Spending Score"] = df["SpendingScore"].mean()
    pop_metrics["Avg Age"] = df["Age"].mean()
    pop_metrics["Avg Income (k)"] = df["AnnualIncome"].mean() / 1000
    if "FamilySize" in df.columns:
        pop_metrics["Avg Family Size"] = df["FamilySize"].mean()
    if "WorkExperience" in df.columns:
        pop_metrics["Avg Work Experience"] = df["WorkExperience"].mean()

    fig = go.Figure(go.Bar(
        x=list(pop_metrics.keys()),
        y=list(pop_metrics.values()),
        marker_color=PALETTE[:len(pop_metrics)],
        text=[f"{v:.1f}" for v in pop_metrics.values()],
        textposition="outside",
        textfont_color="#e4eafa",
    ))
    apply_chart_style(fig, "Key Population Averages", 340)
    st.plotly_chart(fig, use_container_width=True)


# Sidebar and main

def build_sidebar(df):
    st.sidebar.markdown("## Customer Intelligence")
    st.sidebar.markdown(
        "<div style='color:#9aa3bb;font-size:0.78rem;margin-bottom:16px;'>"
        "Kaggle: datascientistanna/customers</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Demographics", "Segmentation", "Relationships", "Insights"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    filtered = df.copy()

    if "Gender" in df.columns:
        genders = sorted(df["Gender"].dropna().unique())
        selected = st.sidebar.multiselect("Gender", genders, default=genders)
        if selected:
            filtered = filtered[filtered["Gender"].isin(selected)]

    if "Age" in df.columns:
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.sidebar.slider("Age range", min_age, max_age, (min_age, max_age))
        filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]

    if "AnnualIncome" in df.columns:
        inc_min = int(df["AnnualIncome"].min())
        inc_max = int(df["AnnualIncome"].max())
        inc_range = st.sidebar.slider("Annual Income ($)", inc_min, inc_max, (inc_min, inc_max), step=1000)
        filtered = filtered[
            (filtered["AnnualIncome"] >= inc_range[0]) &
            (filtered["AnnualIncome"] <= inc_range[1])
        ]

    return page, filtered


def main():
    df = load_data()

    if df is None:
        st.error(
            "Could not find Customers.csv. Download it from "
            "https://www.kaggle.com/datasets/datascientistanna/customers-dataset "
            "and place it in the same folder as this script (or in a ./data subfolder)."
        )
        st.stop()

    page, filtered_df = build_sidebar(df)

    if len(filtered_df) == 0:
        st.warning("No customers match the current filters. Try adjusting the sidebar.")
        st.stop()

    if page == "Overview":
        show_overview(filtered_df)
    elif page == "Demographics":
        show_demographics(filtered_df)
    elif page == "Segmentation":
        show_segmentation(filtered_df)
    elif page == "Relationships":
        show_relationships(filtered_df)
    elif page == "Insights":
        show_insights(filtered_df)


if __name__ == "__main__":
    main()
