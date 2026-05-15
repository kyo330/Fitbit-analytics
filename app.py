"""
Longitudinal Health & Behavior Intelligence Platform
Dataset: FitBit Fitness Tracker Data (Kaggle — arashnic/fitbit)
30 users · 31 days · March–April 2016
Columns: dailyActivity_merged.csv + sleepDay_merged.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitBit Intelligence Platform",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}

.metric-card{background:linear-gradient(135deg,#1e2130,#252a3d);border:1px solid #2d3250;
  border-radius:12px;padding:20px 22px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.3);}
.metric-card .val{font-size:1.9rem;font-weight:700;color:#7c9ef5;line-height:1.1;}
.metric-card .lbl{font-size:.75rem;color:#8b92a8;margin-top:4px;text-transform:uppercase;letter-spacing:.05em;}
.metric-card .dlt{font-size:.8rem;margin-top:5px;}

.section-hdr{font-size:1.35rem;font-weight:700;color:#e2e8f0;border-left:4px solid #7c9ef5;
  padding-left:13px;margin:26px 0 16px;}

.insight-card{background:linear-gradient(135deg,#1a1f35,#1e2540);border:1px solid #2d3a6a;
  border-radius:10px;padding:13px 17px;margin-bottom:9px;font-size:.88rem;color:#c8d4f0;line-height:1.55;}
.insight-card.warn{border-left:3px solid #f6a623;background:linear-gradient(135deg,#1f1a0e,#2a2010);}
.insight-card.good{border-left:3px solid #48bb78;background:linear-gradient(135deg,#0e1f17,#102318);}
.insight-card.bad{border-left:3px solid #fc8181;background:linear-gradient(135deg,#1f0e0e,#231010);}

.platform-title{font-size:1.95rem;font-weight:800;
  background:linear-gradient(135deg,#7c9ef5,#a78bfa,#60d9fa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:3px;}
.platform-sub{color:#4a5568;font-size:.85rem;margin-bottom:22px;}

.pred-box{background:linear-gradient(135deg,#1a2040,#1e2850);border:1px solid #3a4a7a;
  border-radius:12px;padding:20px 26px;text-align:center;}
.pred-box .pval{font-size:2.3rem;font-weight:800;color:#a78bfa;}
.pred-box .plbl{color:#8b92a8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;}

.badge{display:inline-block;background:#7f1d1d;color:#fca5a5;padding:2px 9px;
  border-radius:20px;font-size:.72rem;font-weight:600;margin-left:7px;}

div[data-testid="stMetricValue"]{color:#7c9ef5;font-weight:700;}
.stSidebar{background:#151821!important;}
h1,h2,h3{color:#e2e8f0!important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")

COLORS = dict(
    TotalSteps="#60d9fa", Calories="#f687b3", SedentaryMinutes="#fc8181",
    VeryActiveMinutes="#48bb78", TotalMinutesAsleep="#a78bfa",
    TotalDistance="#f6a623", sleep_efficiency="#7c9ef5",
    anomaly="#ff4444", r7="#ffffff", r14="#ffd700",
)
LABELS = dict(
    TotalSteps="Total Steps", Calories="Calories Burned",
    SedentaryMinutes="Sedentary Minutes", VeryActiveMinutes="Very Active Minutes",
    TotalDistance="Distance (miles)", TotalMinutesAsleep="Sleep (minutes)",
    FairlyActiveMinutes="Fairly Active Minutes", LightlyActiveMinutes="Lightly Active Minutes",
    sleep_efficiency="Sleep Efficiency (%)",
)
PLOT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#a0aec0", font_family="Inter", title_font_color="#e2e8f0",
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#a0aec0"),
    xaxis=dict(gridcolor="#1e2436", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e2436", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=42, b=10), height=360,
)

# ── Data loading & merging ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    act = pd.read_csv(os.path.join(DATA, "dailyActivity_merged.csv"))
    slp = pd.read_csv(os.path.join(DATA, "sleepDay_merged.csv"))

    act["date"] = pd.to_datetime(act["ActivityDate"], format="%m/%d/%Y")
    slp["date"] = pd.to_datetime(slp["SleepDay"].str.split(" ").str[0], format="%m/%d/%Y")

    merged = act.merge(slp[["Id", "date", "TotalMinutesAsleep", "TotalTimeInBed",
                             "TotalSleepRecords"]], on=["Id", "date"], how="left")

    merged["sleep_hours"]     = (merged["TotalMinutesAsleep"] / 60).round(2)
    merged["time_in_bed_hrs"] = (merged["TotalTimeInBed"] / 60).round(2)
    merged["sleep_efficiency"]= (merged["TotalMinutesAsleep"] /
                                  merged["TotalTimeInBed"].replace(0, np.nan) * 100).round(1)
    merged["active_minutes"]  = (merged["VeryActiveMinutes"] + merged["FairlyActiveMinutes"]
                                  + merged["LightlyActiveMinutes"])
    merged["day_of_week"]     = merged["date"].dt.day_name()
    merged["week_num"]        = merged["date"].dt.isocalendar().week.astype(int)
    merged["user_label"]      = "User " + merged["Id"].astype(str).str[-4:]

    return merged.sort_values(["Id", "date"]).reset_index(drop=True)


@st.cache_data
def add_rolling(df):
    df = df.copy()
    cols = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes",
            "TotalMinutesAsleep", "sleep_efficiency", "active_minutes"]
    for uid, grp in df.groupby("Id"):
        idx = grp.index
        for c in cols:
            if c in df.columns:
                df.loc[idx, f"{c}_r7"]  = grp[c].rolling(7,  min_periods=1).mean().round(1).values
                df.loc[idx, f"{c}_r14"] = grp[c].rolling(14, min_periods=1).mean().round(1).values
                df.loc[idx, f"{c}_z"]   = stats.zscore(grp[c].fillna(grp[c].median())).round(3)
    return df


@st.cache_data
def detect_anomalies(df):
    df = df.copy()
    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]
    for uid, grp in df.groupby("Id"):
        idx = grp.index
        X = grp[features].fillna(grp[features].median())
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=100)
        df.loc[idx, "anomaly"]       = iso.fit_predict(Xs)
        df.loc[idx, "anomaly_score"] = -iso.decision_function(Xs)
    return df


@st.cache_data
def train_model(df, target="Calories"):
    feat = ["TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
            "LightlyActiveMinutes", "SedentaryMinutes", "TotalDistance",
            "TotalMinutesAsleep"]
    feat = [f for f in feat if f in df.columns]
    d2 = df.dropna(subset=feat + [target]).copy()
    X, y = d2[feat], d2[target]
    split = int(len(X) * 0.8)
    mdl = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    mdl.fit(X[:split], y[:split])
    preds = mdl.predict(X[split:])
    return dict(
        model=mdl, mae=round(mean_absolute_error(y[split:], preds), 1),
        r2=round(r2_score(y[split:], preds), 3),
        y_test=y[split:].values, y_pred=preds,
        dates=d2["date"].iloc[split:].values,
        importances=pd.Series(mdl.feature_importances_, index=feat).sort_values(ascending=False),
        feat=feat, target=target,
        next_pred=round(mdl.predict(X.iloc[[-1]])[0], 1),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar(df):
    st.sidebar.markdown("## 💪 FitBit Intelligence")
    st.sidebar.markdown(
        "<div style='color:#4a5568;font-size:.76rem;'>Kaggle · arashnic/fitbit<br>"
        "30 users · 31 days · Mar–Apr 2016</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "📌 Overview", "📈 Trends", "🔍 Patterns",
        "🚨 Anomalies", "🔮 Prediction", "🧾 Insights"
    ], label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    users = ["All Users"] + sorted(df["user_label"].unique().tolist())
    sel_user = st.sidebar.selectbox("User", users)

    metrics = st.sidebar.multiselect(
        "Metrics",
        ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes",
         "TotalMinutesAsleep", "sleep_efficiency"],
        default=["TotalSteps", "Calories", "SedentaryMinutes"],
        format_func=lambda x: LABELS.get(x, x),
    )

    show_anom  = st.sidebar.toggle("Show Anomaly Markers", True)
    show_roll  = st.sidebar.toggle("Show Rolling Averages", True)

    # Filter data
    dff = df.copy()
    if sel_user != "All Users":
        dff = dff[dff["user_label"] == sel_user]

    return page, dff, sel_user, metrics, show_anom, show_roll


# ── Page helpers ──────────────────────────────────────────────────────────────
def kpi(val, label, delta="", dcolor="#8b92a8", vcolor="#7c9ef5"):
    return f"""<div class="metric-card">
      <div class="val" style="color:{vcolor}">{val}</div>
      <div class="lbl">{label}</div>
      <div class="dlt" style="color:{dcolor}">{delta}</div>
    </div>"""


def trend_chart(df, metric, show_anom, show_roll, height=340):
    color = COLORS.get(metric, "#7c9ef5")
    label = LABELS.get(metric, metric)
    fig = go.Figure()

    # Per-user lines (thin)
    for uid, grp in df.groupby("Id"):
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp[metric],
            mode="lines", line=dict(color=color, width=1), opacity=0.25,
            showlegend=False,
            hovertemplate=f"User {str(uid)[-4:]}<br>%{{x|%b %d}}: %{{y:.1f}}<extra></extra>",
        ))

    # Population daily mean
    pop = df.groupby("date")[metric].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=pop["date"], y=pop[metric],
        name="Pop avg", line=dict(color=color, width=2.5),
        hovertemplate="<b>Avg</b> %{x|%b %d}: %{y:.1f}<extra></extra>",
    ))

    if show_roll and f"{metric}_r7" in df.columns:
        pr7 = df.groupby("date")[f"{metric}_r7"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=pr7["date"], y=pr7[f"{metric}_r7"],
            name="7-day avg", line=dict(color="#ffffff", width=2, dash="dot"),
        ))

    if show_anom and "anomaly" in df.columns:
        an = df[df["anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=an["date"], y=an[metric], mode="markers",
            marker=dict(color="#ff4444", size=9, symbol="x",
                        line=dict(color="#ff0000", width=2)),
            name="Anomaly",
            hovertemplate="<b>⚠ Anomaly</b><br>%{x|%b %d}: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(**{**PLOT, "height": height,
                         "title": f"{label} — Daily Trend (all users)"})
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_overview(df, sel_user):
    st.markdown('<div class="platform-title">💪 FitBit Health Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="platform-sub">Dataset: <b>Kaggle · arashnic/fitbit</b> · '
        '30 users · 31 days (Mar 12 – Apr 11 2016) · '
        'dailyActivity_merged.csv + sleepDay_merged.csv</div>',
        unsafe_allow_html=True)

    # KPIs
    cols = st.columns(4)
    kpis = [
        (f"{df['TotalSteps'].mean():,.0f}", "Avg Daily Steps",
         f"{df['TotalSteps'].median():,.0f} median", "#60d9fa"),
        (f"{df['Calories'].mean():,.0f}", "Avg Calories/Day",
         f"max {df['Calories'].max():,}", "#f687b3"),
        (f"{df['sleep_hours'].mean():.1f}h", "Avg Sleep",
         f"{df['sleep_hours'].dropna().median():.1f}h median", "#a78bfa"),
        (f"{df['SedentaryMinutes'].mean():.0f}", "Avg Sedentary Mins",
         f"{df['SedentaryMinutes'].mean()/60:.1f}h/day", "#fc8181"),
    ]
    for col, (v, l, d, c) in zip(cols, kpis):
        with col:
            st.markdown(kpi(v, l, d, c, c), unsafe_allow_html=True)

    st.markdown("")
    cols2 = st.columns(4)
    kpis2 = [
        (f"{df['Id'].nunique()}", "Unique Users", "in dataset", "#48bb78"),
        (f"{len(df)}", "Total Records", f"{df['date'].nunique()} days", "#f6a623"),
        (f"{df['VeryActiveMinutes'].mean():.0f}", "Avg Very Active Mins",
         "per day", "#7c9ef5"),
        (f"{df['sleep_efficiency'].mean():.1f}%", "Avg Sleep Efficiency",
         "min asleep / in bed", "#60d9fa"),
    ]
    for col, (v, l, d, c) in zip(cols2, kpis2):
        with col:
            st.markdown(kpi(v, l, d, c, c), unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Data Preview</div>', unsafe_allow_html=True)
    show_cols = ["user_label", "date", "TotalSteps", "TotalDistance", "VeryActiveMinutes",
                 "SedentaryMinutes", "Calories", "sleep_hours", "sleep_efficiency"]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols].head(20), use_container_width=True, height=310)

    st.markdown('<div class="section-hdr">Dataset Summary Statistics</div>', unsafe_allow_html=True)
    stat_cols = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes",
                 "TotalDistance", "sleep_hours", "sleep_efficiency"]
    stat_cols = [c for c in stat_cols if c in df.columns]
    st.dataframe(df[stat_cols].describe().round(2), use_container_width=True)

    # Quick distribution chart
    st.markdown('<div class="section-hdr">Steps Distribution Across All Users</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    for uid, grp in df.groupby("user_label"):
        fig.add_trace(go.Box(y=grp["TotalSteps"], name=uid,
                             marker_color="#60d9fa", line_color="#7c9ef5",
                             showlegend=False, boxmean=True))
    fig.update_layout(**{**PLOT, "height": 380,
                         "title": "Daily Steps Distribution — Per User",
                         "xaxis_tickangle": -45})
    st.plotly_chart(fig, use_container_width=True)


def page_trends(df, metrics, show_anom, show_roll):
    st.markdown('<div class="section-hdr">📈 Behavioral Trends Over Time</div>',
                unsafe_allow_html=True)

    if not metrics:
        st.warning("Pick at least one metric in the sidebar.")
        return

    for m in metrics:
        st.plotly_chart(trend_chart(df, m, show_anom, show_roll), use_container_width=True)

    # Normalized overlay
    st.markdown('<div class="section-hdr">Normalized Multi-Metric Overlay</div>',
                unsafe_allow_html=True)
    pop = df.groupby("date")[metrics].mean()
    fig = go.Figure()
    for m in metrics:
        s = pop[m]
        norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        fig.add_trace(go.Scatter(x=pop.index, y=norm,
                                 name=LABELS.get(m, m),
                                 line=dict(color=COLORS.get(m, "#7c9ef5"), width=2)))
    fig.update_layout(**{**PLOT, "height": 320, "title": "Population Avg — Normalized (0–1)"})
    st.plotly_chart(fig, use_container_width=True)


def page_patterns(df):
    st.markdown('<div class="section-hdr">🔍 Behavioral Pattern Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📅 Day-of-Week", "👤 Per-User Profile", "🔗 Correlations", "🏃 Activity Mix"])

    # ── Day-of-week
    with tab1:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df.groupby("day_of_week")[
            ["TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes","sleep_hours"]
        ].mean().round(1).reindex(dow_order)

        fig = make_subplots(2, 2, subplot_titles=(
            "Steps by Day", "Calories by Day",
            "Sedentary Mins by Day", "Very Active Mins by Day"))
        pairs = [("TotalSteps","#60d9fa",1,1),("Calories","#f687b3",1,2),
                 ("SedentaryMinutes","#fc8181",2,1),("VeryActiveMinutes","#48bb78",2,2)]
        for col, color, r, c in pairs:
            fig.add_trace(go.Bar(x=dow.index, y=dow[col], marker_color=color,
                                 showlegend=False), row=r, col=c)
        fig.update_layout(**{**PLOT, "height": 520, "barmode": "group"})
        fig.update_annotations(font_color="#c0c8e0")
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-user profile
    with tab2:
        user_profile = df.groupby("user_label").agg(
            avg_steps=("TotalSteps","mean"),
            avg_calories=("Calories","mean"),
            avg_sedentary=("SedentaryMinutes","mean"),
            avg_very_active=("VeryActiveMinutes","mean"),
            avg_sleep=("sleep_hours","mean"),
            days=("date","nunique"),
        ).round(1).reset_index()

        st.dataframe(user_profile, use_container_width=True, height=380)

        # Scatter: steps vs calories
        fig2 = px.scatter(user_profile, x="avg_steps", y="avg_calories",
                          text="user_label", size="avg_very_active",
                          color="avg_sleep", color_continuous_scale="Viridis",
                          labels={"avg_steps":"Avg Steps","avg_calories":"Avg Calories",
                                  "avg_sleep":"Avg Sleep (hrs)"},
                          title="User Profile: Steps vs Calories (size=active mins, color=sleep)")
        fig2.update_traces(textposition="top center", textfont_size=9)
        fig2.update_layout(**{**PLOT, "height": 400})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Correlations
    with tab3:
        corr_cols = ["TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes",
                     "TotalDistance","sleep_hours","sleep_efficiency","active_minutes"]
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr = df[corr_cols].corr().round(3)
        lbls = [LABELS.get(c, c).replace(" ","<br>") for c in corr_cols]

        fig = go.Figure(go.Heatmap(
            z=corr.values, x=lbls, y=lbls,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont=dict(size=9, color="white"),
            colorbar=dict(thickness=13, tickfont_color="#a0aec0"),
        ))
        fig.update_layout(**{**PLOT, "height": 490, "title": "Pearson Correlation Matrix"})
        st.plotly_chart(fig, use_container_width=True)

        # Key scatter relationships
        st.markdown("#### Key Pairwise Relationships")
        s_pairs = [("TotalSteps","Calories","#f687b3"),
                   ("TotalSteps","SedentaryMinutes","#fc8181"),
                   ("sleep_hours","VeryActiveMinutes","#48bb78")]
        s_cols = st.columns(3)
        for col, (x, y, color) in zip(s_cols, s_pairs):
            valid = df[[x, y]].dropna()
            r, p = stats.pearsonr(valid[x], valid[y])
            fig_s = px.scatter(df, x=x, y=y, trendline="ols", opacity=0.45,
                               color_discrete_sequence=[color],
                               labels={x: LABELS.get(x,x), y: LABELS.get(y,y)})
            fig_s.update_layout(**{**PLOT, "height": 300,
                "title": f"{LABELS.get(x,x)[:12]} vs {LABELS.get(y,y)[:12]}<br>"
                         f"<sup>r={r:.3f}, p={p:.2e}</sup>"})
            with col:
                st.plotly_chart(fig_s, use_container_width=True)

    # ── Activity mix
    with tab4:
        st.markdown("#### Population-Level Activity Time Breakdown")
        mix = df[["VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes",
                  "SedentaryMinutes"]].mean().round(1)
        labels = ["Very Active","Fairly Active","Lightly Active","Sedentary"]
        colors = ["#48bb78","#f6a623","#60d9fa","#fc8181"]
        fig = go.Figure(go.Pie(labels=labels, values=mix.values,
                               marker_colors=colors, hole=0.45,
                               textinfo="label+percent"))
        fig.update_layout(**{**PLOT, "height": 400,
                              "title": "Avg Daily Minutes by Activity Level"})
        st.plotly_chart(fig, use_container_width=True)

        # Stacked bar per day-of-week
        dow = df.groupby("day_of_week")[
            ["VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"]
        ].mean().round(1).reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        fig2 = go.Figure()
        for col, color, label in zip(
            ["VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"],
            ["#48bb78","#f6a623","#60d9fa","#fc8181"],
            ["Very Active","Fairly Active","Lightly Active","Sedentary"]
        ):
            fig2.add_trace(go.Bar(x=dow.index, y=dow[col], name=label,
                                  marker_color=color))
        fig2.update_layout(**{**PLOT, "height": 360, "barmode": "stack",
                               "title": "Activity Mix by Day of Week"})
        st.plotly_chart(fig2, use_container_width=True)


def page_anomalies(df):
    st.markdown('<div class="section-hdr">🚨 Anomaly Detection</div>', unsafe_allow_html=True)

    anom = df[df["anomaly"] == -1]
    n = len(anom)
    pct = n / len(df) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi(n, "Anomalous Records", f"{pct:.1f}% of dataset", "#fc8181", "#fc8181"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Isolation Forest", "Detection Method",
                        "per-user, 7% contamination", "#f6a623", "#f6a623"),
                    unsafe_allow_html=True)
    with c3:
        hi = len(anom[anom["anomaly_score"] > anom["anomaly_score"].quantile(0.75)])
        st.markdown(kpi(hi, "High-Severity", "top 25% anomaly score", "#fc8181", "#fc8181"),
                    unsafe_allow_html=True)

    st.markdown("")

    # Anomaly score timeline
    pop_score = df.groupby("date")["anomaly_score"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pop_score["date"], y=pop_score["anomaly_score"],
                             fill="tozeroy", fillcolor="rgba(252,129,129,.08)",
                             line=dict(color="#fc8181", width=1.5),
                             name="Avg Anomaly Score"))
    fig.update_layout(**{**PLOT, "height": 280, "title": "Population Anomaly Score Over Time"})
    st.plotly_chart(fig, use_container_width=True)

    # Per-metric scatter with anomalies highlighted
    st.markdown("#### Anomalous Days by Metric")
    anom_metrics = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]
    fig2 = make_subplots(2, 2, subplot_titles=[LABELS[m] for m in anom_metrics])
    pos = [(1,1),(1,2),(2,1),(2,2)]
    for m, (r, c) in zip(anom_metrics, pos):
        color = COLORS.get(m, "#7c9ef5")
        normal = df[df["anomaly"] != -1]
        fig2.add_trace(go.Scatter(x=normal["date"], y=normal[m], mode="lines",
                                  line=dict(color=color, width=1), opacity=0.4,
                                  showlegend=False), row=r, col=c)
        fig2.add_trace(go.Scatter(x=anom["date"], y=anom[m], mode="markers",
                                  marker=dict(color="#ff4444", size=8, symbol="x"),
                                  showlegend=(r==1 and c==1), name="Anomaly"), row=r, col=c)
    fig2.update_layout(**{**PLOT, "height": 520})
    fig2.update_annotations(font_color="#c0c8e0")
    st.plotly_chart(fig2, use_container_width=True)

    # Anomaly table
    st.markdown("#### Top Anomalous Records")
    show = anom[["user_label","date","TotalSteps","Calories","SedentaryMinutes",
                 "VeryActiveMinutes","anomaly_score"]].copy()
    show["anomaly_score"] = show["anomaly_score"].round(4)
    show = show.sort_values("anomaly_score", ascending=False).head(30)
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%b %d, %Y")
    st.dataframe(show, use_container_width=True, height=360)


def page_prediction(df):
    st.markdown('<div class="section-hdr">🔮 Behavioral Forecasting</div>', unsafe_allow_html=True)

    target_opts = {"Calories": "Calories Burned", "TotalSteps": "Total Steps",
                   "SedentaryMinutes": "Sedentary Minutes",
                   "VeryActiveMinutes": "Very Active Minutes"}
    target = st.selectbox("Predict:", list(target_opts.keys()),
                          format_func=lambda x: target_opts[x])

    with st.spinner("Training Random Forest…"):
        res = train_model(df, target)

    c1, c2, c3 = st.columns(3)
    units = {"Calories":"kcal","TotalSteps":"steps",
             "SedentaryMinutes":"mins","VeryActiveMinutes":"mins"}
    with c1:
        st.markdown(f"""<div class="pred-box">
          <div class="plbl">Next-Record Prediction</div>
          <div class="pval">{res['next_pred']:,.0f}</div>
          <div class="plbl" style="color:#60d9fa;margin-top:4px">{units.get(target,'')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(kpi(f"{res['mae']:,.1f}", "Mean Absolute Error",
                        f"on {target_opts[target]}", "#48bb78", "#48bb78"),
                    unsafe_allow_html=True)
    with c3:
        rc = "#48bb78" if res["r2"] > 0.5 else "#f6a623"
        st.markdown(kpi(f"{res['r2']:.3f}", "R² Score", "explained variance", rc, rc),
                    unsafe_allow_html=True)

    st.markdown("")
    ca, cb = st.columns([3, 2])
    with ca:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["dates"], y=res["y_test"],
                                 name="Actual",
                                 line=dict(color=COLORS.get(target,"#7c9ef5"), width=2)))
        fig.add_trace(go.Scatter(x=res["dates"], y=res["y_pred"],
                                 name="Predicted",
                                 line=dict(color="#ffd700", width=2, dash="dot")))
        fig.update_layout(**{**PLOT, "height": 350,
                              "title": f"{target_opts[target]} — Actual vs Predicted"})
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        imp = res["importances"].head(7)
        fig2 = go.Figure(go.Bar(x=imp.values, y=[LABELS.get(i,i) for i in imp.index],
                                orientation="h", marker_color="#7c9ef5"))
        fig2.update_layout(**{**PLOT, "height": 350, "title": "Feature Importances"})
        st.plotly_chart(fig2, use_container_width=True)

    resid = res["y_test"] - res["y_pred"]
    fig3 = go.Figure(go.Histogram(x=resid, nbinsx=28,
                                   marker_color="#a78bfa", opacity=0.8))
    fig3.update_layout(**{**PLOT, "height": 260, "title": "Residuals Distribution"})
    st.plotly_chart(fig3, use_container_width=True)


def page_insights(df):
    st.markdown('<div class="section-hdr">🧾 Automated Behavioral Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;margin-top:-12px;margin-bottom:20px;">'
                'Data-driven interpretations from the FitBit population dataset</p>',
                unsafe_allow_html=True)

    insights = []

    # Steps
    avg_steps = df["TotalSteps"].mean()
    who_rec   = 10000
    pct_who   = (df["TotalSteps"] >= who_rec).mean() * 100
    delta_who = ((avg_steps - who_rec) / who_rec) * 100
    cls = "good" if avg_steps >= who_rec else "bad"
    insights.append((cls,
        f"🚶 The population averages <b>{avg_steps:,.0f} steps/day</b> — "
        f"{'above' if avg_steps >= who_rec else 'below'} the WHO-recommended 10,000 "
        f"by {abs(delta_who):.1f}%. Only <b>{pct_who:.1f}%</b> of recorded days hit 10K+ steps."))

    # Sedentary
    avg_sed_h = df["SedentaryMinutes"].mean() / 60
    insights.append(("bad" if avg_sed_h > 10 else "warn",
        f"🪑 Users average <b>{avg_sed_h:.1f} sedentary hours/day</b> "
        f"({df['SedentaryMinutes'].mean():.0f} mins). "
        f"{'This exceeds health guidelines — prolonged sitting is linked to metabolic risk.' if avg_sed_h > 10 else 'Sedentary time is within moderate range.'}"))

    # Sleep
    avg_sleep = df["sleep_hours"].mean()
    sleep_rec = 7.0
    if not np.isnan(avg_sleep):
        cls2 = "good" if avg_sleep >= sleep_rec else "warn"
        insights.append((cls2,
            f"🛌 Average sleep is <b>{avg_sleep:.2f} hours/night</b> "
            f"({'meets' if avg_sleep >= sleep_rec else 'below'} the 7h minimum recommendation). "
            f"Sleep efficiency averages <b>{df['sleep_efficiency'].mean():.1f}%</b> "
            f"(time asleep ÷ time in bed)."))

    # Correlation: steps ↔ calories
    r_sc, p_sc = stats.pearsonr(df["TotalSteps"].dropna(), df["Calories"].dropna())
    insights.append(("good",
        f"🔗 <b>Steps ↔ Calories</b>: r = {r_sc:.3f} (p={p_sc:.2e}). "
        f"Strong positive relationship — every additional 1,000 steps is associated with "
        f"~{(r_sc * df['Calories'].std() / (df['TotalSteps'].std() / 1000)):.0f} extra calories burned."))

    # Sedentary vs steps correlation
    r_sed, _ = stats.pearsonr(
        df[["TotalSteps","SedentaryMinutes"]].dropna()["TotalSteps"],
        df[["TotalSteps","SedentaryMinutes"]].dropna()["SedentaryMinutes"])
    insights.append(("warn" if r_sed < -0.2 else "neutral",
        f"📊 <b>Steps ↔ Sedentary Time</b>: r = {r_sed:.3f}. "
        f"{'Negative correlation confirms that more active users spend less time sedentary.' if r_sed < -0.2 else 'Weak relationship between step count and sedentary time in this cohort.'}"))

    # Best vs worst user
    up = df.groupby("user_label")["TotalSteps"].mean().sort_values()
    insights.append(("good",
        f"🏆 Most active user: <b>{up.index[-1]}</b> — avg {up.iloc[-1]:,.0f} steps/day. "
        f"Least active: <b>{up.index[0]}</b> — avg {up.iloc[0]:,.0f} steps/day. "
        f"A <b>{(up.iloc[-1]/up.iloc[0]):.1f}×</b> difference across the cohort."))

    # Day-of-week best
    dow = df.groupby("day_of_week")["TotalSteps"].mean()
    best_day = dow.idxmax(); worst_day = dow.idxmin()
    insights.append(("neutral",
        f"📅 Most active day: <b>{best_day}</b> ({dow[best_day]:,.0f} avg steps). "
        f"Least active: <b>{worst_day}</b> ({dow[worst_day]:,.0f} avg steps). "
        f"Weekend patterns show {'increased' if dow.get('Saturday',0) > dow.get('Wednesday',0) else 'reduced'} activity."))

    # Anomaly insight
    n_anom = (df["anomaly"] == -1).sum()
    insights.append(("warn",
        f"🚨 <b>{n_anom} anomalous records</b> detected across {len(df)} entries "
        f"({n_anom/len(df)*100:.1f}%) using Isolation Forest per-user. "
        f"These flag unusual combinations of low activity, high sedentary time, "
        f"or abnormal calorie burn for that individual."))

    # Very active minutes
    avg_vam = df["VeryActiveMinutes"].mean()
    insights.append(("good" if avg_vam >= 21 else "warn",
        f"💪 Average <b>{avg_vam:.1f} very active minutes/day</b>. "
        f"WHO recommends 150 mins of moderate or 75 mins of vigorous activity per week — "
        f"{'this cohort exceeds' if avg_vam*7 >= 75 else 'this cohort does not meet'} the vigorous threshold "
        f"({avg_vam*7:.0f} weekly mins vs 75 recommended)."))

    css = {"good":"good","bad":"bad","warn":"warn","neutral":""}
    for cls, txt in insights:
        st.markdown(f'<div class="insight-card {css.get(cls,"")}">{txt}</div>',
                    unsafe_allow_html=True)

    # Weekly rollup sparklines
    st.markdown("---")
    st.markdown("#### Population Trend — Daily Averages")
    pop_daily = df.groupby("date")[
        ["TotalSteps","Calories","SedentaryMinutes","VeryActiveMinutes"]
    ].mean().reset_index()

    fig = make_subplots(1, 4, subplot_titles=["Steps","Calories","Sedentary Mins","Very Active Mins"])
    for i, (col, color) in enumerate([
        ("TotalSteps","#60d9fa"),("Calories","#f687b3"),
        ("SedentaryMinutes","#fc8181"),("VeryActiveMinutes","#48bb78")
    ], 1):
        fig.add_trace(go.Scatter(x=pop_daily["date"], y=pop_daily[col],
                                 fill="tozeroy", line=dict(color=color, width=2),
                                 showlegend=False), row=1, col=i)
    fig.update_layout(**{**PLOT, "height": 240})
    fig.update_annotations(font_color="#9aa8c8", font_size=10)
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df_raw  = load_data()
    df_roll = add_rolling(df_raw)
    df_full = detect_anomalies(df_roll)

    page, dff, sel_user, metrics, show_anom, show_roll = sidebar(df_full)

    if len(dff) == 0:
        st.error("No data for selected filters.")
        return

    if   page == "📌 Overview":   page_overview(dff, sel_user)
    elif page == "📈 Trends":     page_trends(dff, metrics, show_anom, show_roll)
    elif page == "🔍 Patterns":   page_patterns(dff)
    elif page == "🚨 Anomalies":  page_anomalies(dff)
    elif page == "🔮 Prediction": page_prediction(dff)
    elif page == "🧾 Insights":   page_insights(dff)


if __name__ == "__main__":
    main()
