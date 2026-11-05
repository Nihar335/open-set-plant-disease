"""Weather for the user's state (Open-Meteo API, free, no key) and disease-weather favourability.

Only the state capital's coordinates are sent to api.open-meteo.com; no image or personal data leaves the machine.
If the API cannot be reached the diagnosis continues without weather (factor 1.0 for every disease).
"""
from __future__ import annotations

import time

import requests

API = "https://api.open-meteo.com/v1/forecast"
_CACHE: dict = {}
CACHE_SECONDS = 1800


def fetch(lat: float, lon: float) -> dict | None:
    key = (round(lat, 2), round(lon, 2))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    params = {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata",
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
        "past_days": 7, "forecast_days": 3,
    }
    try:
        r = requests.get(API, params=params, timeout=8)
        r.raise_for_status()
        raw = r.json()
    except (requests.RequestException, ValueError):
        return None
    d = raw["daily"]
    past = slice(0, 7)  # the 7 days before today
    tmean = [(a + b) / 2 for a, b in zip(d["temperature_2m_max"][past], d["temperature_2m_min"][past])]
    rh = [v for v in d["relative_humidity_2m_mean"][past] if v is not None]
    rain = d["precipitation_sum"][past]
    summary = {
        "current_temp_c": raw["current"]["temperature_2m"],
        "current_humidity_pct": raw["current"]["relative_humidity_2m"],
        "current_precip_mm": raw["current"]["precipitation"],
        "week_mean_temp_c": round(sum(tmean) / len(tmean), 1),
        "week_mean_humidity_pct": round(sum(rh) / len(rh), 1) if rh else None,
        "week_rain_mm": round(sum(v or 0 for v in rain), 1),
        "week_rain_days": sum(1 for v in rain if (v or 0) >= 1.0),
        "forecast_rain_mm_3d": round(sum(v or 0 for v in d["precipitation_sum"][7:]), 1),
        "daily": [{"date": t, "tmax": a, "tmin": b, "rain_mm": p, "rh": h} for t, a, b, p, h in zip(
            d["time"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_sum"],
            d["relative_humidity_2m_mean"])],
        "source": "Open-Meteo (open-meteo.com)",
    }
    _CACHE[key] = (time.time(), summary)
    return summary


GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"


def geocode(place: str, state_name: str) -> dict | None:
    """Find a district / town / village inside the chosen state (Open-Meteo geocoding, India only).

    Returns {name, district, state, lat, lon} or None. Only the place name is sent.
    """
    place = (place or "").strip()
    if not place:
        return None
    try:
        r = requests.get(GEOCODE_API, params={"name": place, "count": 10, "language": "en", "format": "json",
                                              "countryCode": "IN"}, timeout=8)
        r.raise_for_status()
        results = r.json().get("results") or []
    except (requests.RequestException, ValueError):
        return None
    want = state_name.lower()
    in_state = [x for x in results if (x.get("admin1") or "").lower() == want]
    if not in_state:
        return None
    best = max(in_state, key=lambda x: x.get("population") or 0)
    return {"name": best["name"], "district": best.get("admin2"), "state": best.get("admin1"),
            "lat": round(best["latitude"], 4), "lon": round(best["longitude"], 4)}


def favourability(spec: dict | None, w: dict | None) -> dict | None:
    """Score 0..1 for how favourable the last week's weather was for a disease, with a short explanation."""
    if not spec or not w:
        return None
    lo, hi = spec["temp"]
    t = w["week_mean_temp_c"]
    gap = 0 if lo <= t <= hi else min(abs(t - lo), abs(t - hi))
    s_temp = max(0.0, 1 - gap / 6)

    rh = w["week_mean_humidity_pct"] if w["week_mean_humidity_pct"] is not None else w["current_humidity_pct"]
    s_rh = {"high": 1.0 if rh >= 80 else 0.5 if rh >= 60 else 0.1,
            "moderate": 1.0 if 60 <= rh <= 85 else 0.5,
            "low": 1.0 if rh < 60 else 0.5 if rh < 75 else 0.2,
            "any": 1.0}[spec["humidity"]]

    wet_days = w["week_rain_days"] + (1 if w["forecast_rain_mm_3d"] >= 2 else 0)
    s_rain = {"wet": 1.0 if wet_days >= 2 else 0.6 if wet_days == 1 else 0.2,
              "dry": 1.0 if wet_days == 0 else 0.6 if wet_days <= 2 else 0.2,
              "any": 1.0}[spec["rain"]]

    score = (s_temp + s_rh + s_rain) / 3
    level = "high" if score >= 0.75 else "moderate" if score >= 0.45 else "low"
    why = (f"favours {lo}–{hi} °C (last week {t} °C), {spec['humidity']} humidity (last week {rh}%), "
           f"{spec['rain']} weather ({w['week_rain_days']} rain days last week)")
    return {"score": round(score, 2), "level": level, "explanation": why}


def factor(fav: dict | None) -> float:
    """Bounded prior multiplier 0.8..1.2 on a disease's score. Hand-set (no weather-labelled photos exist to learn it)."""
    return 1.0 if fav is None else 0.8 + 0.4 * fav["score"]
