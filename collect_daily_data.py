"""
Daily Data Collection Script
Pakistan Disaster Forecasting System

This script runs automatically every day via GitHub Actions.
It fetches live weather data for all 8 Pakistani cities,
scores sentiment using VADER, and appends results to the
30-day tracking CSV file.

RoBERTa is not used here because GitHub Actions free tier
does not have enough memory to run transformer models.
VADER alone is sufficient for daily automated collection.
RoBERTa is used in the full Colab notebook for deeper analysis.
"""

import os
import requests
import numpy as np
import pandas as pd
import feedparser
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ── Configuration ─────────────────────────────────────────────

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GDACS_RSS_URL  = "https://www.gdacs.org/xml/rss.xml"
TRACKING_FILE  = "data/30day_wri_tracking.csv"
DASHBOARD_FILE = "data/daily_dashboard.png"
ALERTS_FILE    = "data/daily_alerts.csv"

TARGET_CITIES = [
    {"name": "Karachi",    "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore",     "lat": 31.5204, "lon": 74.3587},
    {"name": "Peshawar",   "lat": 34.0150, "lon": 71.5249},
    {"name": "Quetta",     "lat": 30.1798, "lon": 66.9750},
    {"name": "Islamabad",  "lat": 33.6844, "lon": 73.0479},
    {"name": "Hyderabad",  "lat": 25.3960, "lon": 68.3578},
    {"name": "Multan",     "lat": 30.1575, "lon": 71.5249},
    {"name": "Rawalpindi", "lat": 33.5651, "lon": 73.0169},
]

DISASTER_LEXICON = {
    "evacuate": -3.5, "evacuation": -3.5, "trapped": -4.0,
    "stranded": -3.5, "submerged": -3.5, "catastrophic": -4.0,
    "life-threatening": -4.5, "fatalities": -4.0, "casualties": -3.5,
    "cyclone": -3.0, "landslide": -3.5, "displaced": -3.0,
    "mayday": -5.0, "emergency": -3.5, "collapsed": -3.5,
    "inundated": -3.5, "monsoon": -2.0, "rescue": -2.5,
    "missing": -3.0, "destroyed": -4.0, "flooding": -3.5,
    "devastated": -4.0,
}


# ── Weather Functions ──────────────────────────────────────────

def fetch_weather(city):
    params = {
        "latitude"     : city["lat"],
        "longitude"    : city["lon"],
        "current"      : [
            "temperature_2m", "windspeed_10m", "precipitation",
            "relativehumidity_2m", "surface_pressure", "weathercode"
        ],
        "timezone"     : "Asia/Karachi",
        "forecast_days": 1
    }

    condition_map = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
        3: "Overcast", 45: "Foggy", 51: "Light drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        80: "Showers", 95: "Thunderstorm", 99: "Severe thunderstorm"
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        response.raise_for_status()
        current = response.json().get("current", {})
        wcode   = current.get("weathercode", 0)

        return {
            "city"      : city["name"],
            "temp_c"    : round(current.get("temperature_2m",    0.0), 1),
            "wind_kmh"  : round(current.get("windspeed_10m",     0.0), 1),
            "rain_mm"   : round(current.get("precipitation",     0.0), 2),
            "humidity"  : int(current.get("relativehumidity_2m", 0)),
            "pressure"  : round(current.get("surface_pressure",  0.0), 1),
            "condition" : condition_map.get(wcode, f"Code {wcode}"),
            "status"    : "success"
        }

    except Exception as e:
        print(f"  Weather error for {city['name']}: {e}")
        return {
            "city": city["name"], "temp_c": None, "wind_kmh": None,
            "rain_mm": None, "humidity": None, "pressure": None,
            "condition": "unavailable", "status": "error"
        }


# ── WRI Functions ──────────────────────────────────────────────

def calculate_baseline_wri(wind_kmh, rain_mm, w1=0.6, w2=0.4):
    if wind_kmh is None or rain_mm is None:
        return 0.0
    norm_wind = min(wind_kmh / 200.0, 1.0)
    norm_rain = min(rain_mm  / 150.0, 1.0)
    return round((norm_wind * w1 + norm_rain * w2) * 100, 2)


def classify_risk(score):
    if   score < 20: return "LOW"
    elif score < 40: return "MODERATE"
    elif score < 65: return "HIGH"
    else:            return "CRITICAL"


# ── NLP Functions ──────────────────────────────────────────────

def fetch_gdacs_texts():
    feed    = feedparser.parse(GDACS_RSS_URL)
    records = []
    KEEP    = ["flood","cyclone","storm","tropical","hurricane","typhoon","rain"]

    for entry in feed.entries:
        title    = entry.get("title",   "")
        summary  = entry.get("summary", "")
        combined = f"{title} {summary}".lower()
        if any(word in combined for word in KEEP):
            records.append(f"{title}. {summary}".strip()[:400])

    return records


def compute_panic_score(vader, texts):
    if not texts:
        return 0.5
    scores = []
    for text in texts:
        compound = vader.polarity_scores(str(text))["compound"]
        scores.append((1 - compound) / 2)
    return round(float(np.mean(scores)), 4)


# ── Alert Functions ────────────────────────────────────────────

def generate_alert(city, dynamic_wri, dynamic_risk, panic):
    if dynamic_wri >= 80 or dynamic_risk == "CRITICAL":
        return "EMERGENCY"
    elif dynamic_risk == "HIGH" and panic > 0.70:
        return "WARNING"
    elif dynamic_risk == "MODERATE" and panic > 0.60:
        return "ELEVATED WATCH"
    elif dynamic_risk == "MODERATE":
        return "WATCH"
    elif dynamic_risk == "LOW" and panic > 0.65:
        return "ADVISORY"
    else:
        return "NORMAL"


# ── Visualisation ──────────────────────────────────────────────

def generate_dashboard(history_df, today_df, panic_score):
    sns.set_theme(style="whitegrid", palette="muted")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Pakistan Disaster Forecasting — 30-Day Tracking Dashboard\n"
        "Hybrid NLP-Reflex System",
        fontsize=14, fontweight="bold", y=0.98
    )

    cities = today_df["city"].tolist()

    # Plot 1: Today's Baseline vs Dynamic WRI
    ax1 = fig.add_subplot(2, 3, 1)
    x, w = np.arange(len(today_df)), 0.35
    ax1.bar(x - w/2, today_df["baseline_wri"], w,
            label="Baseline", color="#3498db", alpha=0.85)
    ax1.bar(x + w/2, today_df["dynamic_wri"],  w,
            label="Dynamic",  color="#e74c3c", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cities, rotation=45, ha="right", fontsize=8)
    ax1.axhline(40, color="orange", linestyle="--", alpha=0.5)
    ax1.axhline(65, color="red",    linestyle="--", alpha=0.5)
    ax1.set_ylabel("WRI Score")
    ax1.set_title("Today: Baseline vs Dynamic WRI", fontweight="bold")
    ax1.legend(fontsize=8)

    # Plot 2: Today's panic score
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar(cities, [panic_score] * len(cities),
            color="#e74c3c", alpha=0.8, edgecolor="white")
    ax2.set_xticklabels(cities, rotation=45, ha="right", fontsize=8)
    ax2.axhline(0.65, color="red",    linestyle="--", alpha=0.6,
                label="High threshold")
    ax2.axhline(0.45, color="orange", linestyle="--", alpha=0.6,
                label="Moderate threshold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("VADER Panic Score")
    ax2.set_title("Today: NLP Panic Score", fontweight="bold")
    ax2.legend(fontsize=8)

    # Plots 3-8: 30-day trend per city (one per city)
    if len(history_df) > 1:
        history_df["date"] = pd.to_datetime(history_df["date"])

        for i, city in enumerate(cities):
            ax = fig.add_subplot(2, 3, 3 + i) if i < 4 else None
            if ax is None:
                break
            city_data = history_df[
                history_df["city"] == city
            ].sort_values("date")

            if len(city_data) < 2:
                continue

            ax.plot(city_data["date"], city_data["baseline_wri"],
                    color="#3498db", linewidth=2,
                    label="Baseline", linestyle="--")
            ax.plot(city_data["date"], city_data["dynamic_wri"],
                    color="#e74c3c", linewidth=2,
                    label="Dynamic")
            ax.fill_between(
                city_data["date"],
                city_data["baseline_wri"],
                city_data["dynamic_wri"],
                alpha=0.15, color="#e74c3c"
            )
            ax.set_title(f"{city} — 30-Day Trend", fontweight="bold")
            ax.set_ylabel("WRI Score")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(DASHBOARD_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved to {DASHBOARD_FILE}")


# ── Main Pipeline ──────────────────────────────────────────────

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"Running daily collection for {today}")
    print("=" * 55)

    os.makedirs("data", exist_ok=True)

    # Step 1: Fetch weather
    print("Fetching live weather from Open-Meteo...")
    weather_records = []
    for city in TARGET_CITIES:
        record = fetch_weather(city)
        weather_records.append(record)
        print(f"  {record['city']:12s}  "
              f"Wind {record.get('wind_kmh','N/A')} km/h  "
              f"Rain {record.get('rain_mm','N/A')} mm  "
              f"{record['condition']}")

    weather_df = pd.DataFrame(weather_records)

    # Step 2: Calculate WRI
    weather_df["baseline_wri"] = weather_df.apply(
        lambda r: calculate_baseline_wri(r["wind_kmh"], r["rain_mm"]),
        axis=1
    )
    weather_df["baseline_risk"] = weather_df["baseline_wri"].apply(classify_risk)

    # Step 3: Compute panic score from GDACS texts using VADER
    print("\nComputing NLP panic score from GDACS alerts...")
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(DISASTER_LEXICON)
    gdacs_texts  = fetch_gdacs_texts()
    panic_score  = compute_panic_score(vader, gdacs_texts)
    print(f"  GDACS texts analysed : {len(gdacs_texts)}")
    print(f"  VADER panic score    : {panic_score:.4f}")

    weather_df["panic_score"] = panic_score

    # Step 4: Dynamic WRI
    weather_df["multiplier"]  = 1 + (panic_score * 0.5)
    weather_df["dynamic_wri"] = weather_df.apply(
        lambda r: round(
            min(r["baseline_wri"] * r["multiplier"], 100), 2
        ), axis=1
    )
    weather_df["dynamic_risk"] = weather_df["dynamic_wri"].apply(classify_risk)
    weather_df["wri_increase"] = (
        weather_df["dynamic_wri"] - weather_df["baseline_wri"]
    ).round(2)
    weather_df["pct_change"]   = weather_df.apply(
        lambda r: round(
            r["wri_increase"] / max(r["baseline_wri"], 0.01) * 100, 1
        ), axis=1
    )

    # Step 5: Generate alerts
    weather_df["alert"] = weather_df.apply(
        lambda r: generate_alert(
            r["city"], r["dynamic_wri"],
            r["dynamic_risk"], r["panic_score"]
        ), axis=1
    )

    weather_df["date"] = today

    # Step 6: Append to tracking file
    if os.path.exists(TRACKING_FILE):
        existing   = pd.read_csv(TRACKING_FILE)
        # Remove today's data if re-running
        existing   = existing[existing["date"] != today]
        updated    = pd.concat([existing, weather_df], ignore_index=True)
    else:
        updated = weather_df.copy()

    updated.to_csv(TRACKING_FILE, index=False)

    day_number = updated["date"].nunique()
    print(f"\nTracking file updated")
    print(f"  Day number  : {day_number} of 30")
    print(f"  Total rows  : {len(updated)}")

    # Step 7: Save today's alerts
    weather_df[[
        "city", "date", "wind_kmh", "rain_mm",
        "baseline_wri", "baseline_risk",
        "dynamic_wri", "dynamic_risk",
        "panic_score", "alert", "wri_increase"
    ]].to_csv(ALERTS_FILE, index=False)

    # Step 8: Generate dashboard
    print("\nGenerating dashboard...")
    generate_dashboard(updated, weather_df, panic_score)

    # Step 9: Print today's summary
    print(f"\nToday's Summary — {today}")
    print("=" * 70)
    print(f"{'City':12s}  {'Wind':>6s}  {'Rain':>5s}  "
          f"{'Base WRI':>8s}  {'Dyn WRI':>7s}  "
          f"{'Risk':>8s}  {'Alert':>14s}")
    print("-" * 70)
    for _, r in weather_df.iterrows():
        print(f"{str(r['city']):12s}  "
              f"{float(r['wind_kmh'] or 0):6.1f}  "
              f"{float(r['rain_mm'] or 0):5.2f}  "
              f"{float(r['baseline_wri']):8.2f}  "
              f"{float(r['dynamic_wri']):7.2f}  "
              f"{str(r['dynamic_risk']):8s}  "
              f"{str(r['alert']):14s}")

    print(f"\nPanic score : {panic_score:.4f}")
    print(f"Day {day_number} of 30 complete")


if __name__ == "__main__":
    main()
