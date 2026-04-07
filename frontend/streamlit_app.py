"""Streamlit frontend for the flood risk system."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from requests import HTTPError

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BEST_MODEL_NAME = "xgb"

st.set_page_config(page_title="Flood Risk Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(12, 72, 58, 0.25), transparent 30%),
            radial-gradient(circle at top right, rgba(168, 77, 32, 0.18), transparent 26%),
            linear-gradient(180deg, #08111a 0%, #0d1621 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.9rem 1rem;
        border-radius: 18px;
    }
    div.stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: linear-gradient(135deg, #12344a 0%, #0e6b57 100%);
        color: white;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border-color: rgba(255, 255, 255, 0.24);
    }
    .hero-card, .section-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        color: #f6fbff;
    }
    .hero-subtitle {
        color: #c7d8e5;
        font-size: 1rem;
        line-height: 1.6;
    }
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #f6fbff;
        margin-bottom: 0.25rem;
    }
    .section-note {
        color: #b9c9d6;
        font-size: 0.95rem;
        margin-bottom: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Real-Time Flood Risk And Evacuation Dashboard</div>
        <div class="hero-subtitle">
            Search a place in Gurugram, fetch live conditions automatically, and get the safest shelter route
            using the fixed best-performing model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def fetch_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{st.session_state.backend_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Backend is unreachable. Start the FastAPI server or update the API base URL in the sidebar. "
            f"Current URL: {st.session_state.backend_url}"
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"Backend timed out while calling {path}. Check whether the API is running and responsive."
        ) from exc
    except HTTPError as exc:
        message = f"Backend returned {response.status_code} for {path}."
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message) from exc


def backend_health_status() -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{st.session_state.backend_url.rstrip('/')}/api/health",
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "OK":
            return True, "Backend reachable"
        return False, "Backend responded unexpectedly"
    except requests.RequestException:
        return False, "Backend unavailable"


def initialize_state() -> None:
    st.session_state.setdefault("backend_url", BACKEND_URL)
    st.session_state.setdefault("location_query", "")
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("selected_label", "")
    st.session_state.setdefault("selected_location", None)
    st.session_state.setdefault("latest_weather", None)
    st.session_state.setdefault("latest_prediction", None)
    st.session_state.setdefault("latest_route", None)
    st.session_state.setdefault("backend_status", backend_health_status())


def selected_coordinates() -> tuple[float, float]:
    location = st.session_state.get("selected_location")
    if not location:
        raise ValueError("Search and select a location first.")
    return float(location["latitude"]), float(location["longitude"])


def config_panel() -> None:
    with st.sidebar:
        st.header("Backend")
        st.session_state.backend_url = st.text_input(
            "API base URL",
            value=st.session_state.backend_url,
        )
        if st.button("Check Backend Health", use_container_width=True):
            st.session_state.backend_status = backend_health_status()
        is_healthy, backend_message = st.session_state.backend_status
        if is_healthy:
            st.success(backend_message)
        else:
            st.warning(
                f"{backend_message}. If you are running locally, start FastAPI on `http://127.0.0.1:8000`. "
                "If you are using containers, set the correct `BACKEND_URL`."
            )
        st.caption(f"Fixed model in use: {BEST_MODEL_NAME.upper()}")
        if st.button("Check Project Config", use_container_width=True):
            try:
                result = fetch_json("GET", "/api/config-check")
                if result["ok"]:
                    st.success("All required keys and files are configured.")
                else:
                    st.warning("Setup is incomplete.")
                for name, item in result["checks"].items():
                    st.write(f"- {name}: {'OK' if item['ok'] else 'Missing'}")
            except RuntimeError as exc:
                st.error(str(exc))
        st.caption(
            "Model comparison charts and confusion matrices are saved in backend/app/models/reports."
        )


def location_panel() -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">1. Choose Current Location</div>
            <div class="section-note">Search a locality, chowk, sector, or landmark in Gurugram.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([2, 1])

    with left:
        query = st.text_input(
            "Type a location",
            value=st.session_state.location_query,
            placeholder="Sohna Chowk, Badshahpur, Rajiv Chowk, Sector 45...",
        )
        st.session_state.location_query = query

    with right:
        search_clicked = st.button("Search Place", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("Enter a location query before searching.")
            return
        try:
            result = fetch_json(
                "GET",
                f"/api/weather/search-location?query={requests.utils.quote(query)}&size=5",
            )
            st.session_state.search_results = result["results"]
            labels = [item["label"] for item in result["results"]]
            st.session_state.selected_label = labels[0] if labels else ""
            st.session_state.selected_location = result["results"][0] if result["results"] else None
            st.session_state.latest_weather = None
            st.session_state.latest_prediction = None
            st.session_state.latest_route = None
            st.session_state.backend_status = (True, "Backend reachable")
        except RuntimeError as exc:
            st.error(str(exc))

    results = st.session_state.search_results
    if results:
        labels = [item["label"] for item in results]
        selected_label = st.selectbox(
            "Matching places",
            options=labels,
            index=labels.index(st.session_state.selected_label) if st.session_state.selected_label in labels else 0,
        )
        st.session_state.selected_label = selected_label
        st.session_state.selected_location = next(
            item for item in results if item["label"] == selected_label
        )

    location = st.session_state.selected_location
    if location:
        info1, info2, info3 = st.columns(3)
        info1.metric("Latitude", round(float(location["latitude"]), 6))
        info2.metric("Longitude", round(float(location["longitude"]), 6))
        info3.metric("Region", location.get("region") or "Unknown")
        st.caption(f"Selected place: {location['label']}")


def refresh_live_outputs() -> None:
    latitude, longitude = selected_coordinates()
    st.session_state.latest_weather = fetch_json(
        "GET",
        f"/api/weather/current?latitude={latitude}&longitude={longitude}",
    )
    st.session_state.latest_prediction = fetch_json(
        "POST",
        "/api/risk/predict",
        {
            "latitude": latitude,
            "longitude": longitude,
            "model_name": BEST_MODEL_NAME,
        },
    )
    st.session_state.latest_route = fetch_json(
        "POST",
        "/api/routing/evacuate",
        {
            "latitude": latitude,
            "longitude": longitude,
            "max_candidates": 5,
            "model_name": BEST_MODEL_NAME,
        },
    )


def operations_panel() -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">2. Live Risk And Evacuation</div>
            <div class="section-note">Weather, flood risk, and safe routing are fetched automatically in one step.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Run Live Analysis", use_container_width=True):
        try:
            refresh_live_outputs()
            st.session_state.backend_status = (True, "Backend reachable")
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))

    weather_col, risk_col, route_col = st.columns(3)

    with weather_col:
        if st.session_state.latest_weather:
            current = st.session_state.latest_weather["current"]
            st.metric("Temperature", f"{current['temperature_c']} C")
            st.metric("Humidity", f"{current['humidity_pct']} %")
            st.metric("Rainfall", f"{current['rainfall_mm_hr']} mm/hr")
            st.caption(current["condition"])
        else:
            st.info("Run live analysis to fetch current weather automatically.")

    with risk_col:
        if st.session_state.latest_prediction:
            risk_value = st.session_state.latest_prediction["prediction"]
            st.metric("Flood Risk", risk_value)
            st.metric("Confidence", st.session_state.latest_prediction["confidence"])
            st.caption(f"Fixed model: {BEST_MODEL_NAME.upper()}")
        else:
            st.info("Risk will be computed automatically from live weather.")

    with route_col:
        if st.session_state.latest_route:
            best = st.session_state.latest_route["best_route"]
            st.write("Best Shelter")
            st.write(f"**{best['shelter']['name']}**")
            st.metric("Distance", f"{best['distance_km']} km")
            st.metric("ETA", f"{best['duration_minutes']} min")
        else:
            st.info("Shelter route will appear after live analysis.")

    if st.session_state.latest_route:
        best = st.session_state.latest_route["best_route"]
        st.subheader("Directions")
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Shelter</div>
                <div class="section-note"><strong>{best['shelter']['name']}</strong><br>{best['shelter'].get('address', 'Unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for idx, step in enumerate(best["directions"], start=1):
            st.write(f"{idx}. {step}")
        altitude_df = pd.DataFrame(best["altitude_profile"])
        if not altitude_df.empty:
            st.subheader("Altitude Profile")
            st.line_chart(altitude_df["elevation_m"])


initialize_state()
config_panel()
location_panel()
operations_panel()
