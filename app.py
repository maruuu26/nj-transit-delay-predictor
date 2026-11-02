"""Streamlit dashboard for the NJ Transit delay project.

Run from the repo root (after running src/train.py once):
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent / "outputs"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="NJ Transit Delay Risk", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(OUT / "model.joblib")


@st.cache_data
def load_tables():
    rates = pd.read_csv(OUT / "late_rate_by_line_hour_day.csv")
    stations = pd.read_csv(OUT / "stations_by_line.csv")
    info = json.loads((OUT / "metrics.json").read_text())["info"]
    return rates, stations, info


rates, stations, info = load_tables()
model = load_model()

st.title("NJ Transit delay risk")
st.caption(
    "Built from historical NJ Transit stop data (March 2018 to December 2019, community-scraped). "
    "A stop counts as late when the train is 5+ minutes behind schedule. "
    "This shows past patterns. It is not a live tracker."
)

tab_patterns, tab_estimate = st.tabs(["Delay patterns", "Risk estimator"])

with tab_patterns:
    choice = st.selectbox("Line", ["All lines"] + sorted(rates["line"].unique()))
    view = rates if choice == "All lines" else rates[rates["line"] == choice]

    col_a, col_b = st.columns(2)

    by_hour = view.groupby("hour")[["stops", "late_stops"]].sum()
    by_hour["% late"] = 100 * by_hour["late_stops"] / by_hour["stops"]
    with col_a:
        st.subheader("Late stops by hour of day")
        st.bar_chart(by_hour["% late"])

    grid = view.groupby(["hour", "dayofweek"])[["stops", "late_stops"]].sum()
    grid["pct"] = 100 * grid["late_stops"] / grid["stops"]
    grid = grid["pct"].unstack("dayofweek").reindex(range(24)).reindex(columns=range(7))
    with col_b:
        st.subheader("Late stops by hour and day")
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(grid.values, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(range(7), DAYS)
        ax.set_yticks(range(0, 24, 2), range(0, 24, 2))
        ax.set(xlabel="Day of week", ylabel="Scheduled hour")
        fig.colorbar(image, ax=ax, label="% late")
        st.pyplot(fig)

    st.caption(f"Based on {int(view['stops'].sum()):,} stops in the sample.")

with tab_estimate:
    st.write("Pick a stop and a time to see the model's estimated chance of a 5+ minute delay.")
    c1, c2 = st.columns(2)
    line = c1.selectbox("Line", sorted(stations["line"].unique()), key="est_line")
    options = stations.loc[stations["line"] == line, "to_station"].tolist()
    station = c2.selectbox("Station", options)
    c3, c4, c5 = st.columns(3)
    day = c3.selectbox("Day", DAYS, index=1)
    hour = c4.slider("Scheduled hour", 0, 23, 8)
    month = c5.selectbox("Month", MONTHS, index=5)

    day_idx = DAYS.index(day)
    is_weekend = int(day_idx >= 5)
    is_rush = int(not is_weekend and (6 <= hour <= 9 or 16 <= hour <= 19))
    row = pd.DataFrame(
        [{
            "line": line, "to_station": station, "hour": hour, "dayofweek": day_idx,
            "month": MONTHS.index(month) + 1, "is_weekend": is_weekend, "is_rush": is_rush,
        }]
    )
    prob = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated chance of a 5+ minute delay at this stop", f"{prob:.1%}")
    st.caption(
        "This uses only the schedule (line, station, time, day, month). It cannot see weather, "
        "incidents, or what is happening on the track right now."
    )
