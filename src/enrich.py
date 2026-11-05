"""Enrich data/merged.parquet with FREE external features (all real sources, no synthetic data).

    python src/enrich.py            # fetch anything missing into data/external/, write data/enriched.parquet + docs
    python src/enrich.py --refresh  # re-download everything

Each source is cached under data/external/ (gitignored). Feature groups (prefixes):
    cal_  calendar / holidays (python `holidays` package, state level for each metro)
    fest_ major-festival windows (same package) + approximate school / wedding season flags
    crk_  real cricket match dates (Cricsheet): IPL + internationals, by host city
    wxa_  ACTUAL weather on the journey date (Open-Meteo / ERA5)  -> NOT known at booking time: upper bound
    wxl_  CLIMATOLOGICAL weather for the journey date (2015-2021 same-calendar-day mean) -> known at booking time
    fx_   jet fuel, Brent, USD/INR (FRED) on journey date and on scrape (observation) date
    apt_  airport / city static features (OurAirports, OpenFlights, Wikidata/Census 2011)
Observation (scrape) date = journey_date - days_left: 2022-02-11 (assumed in build_dataset.py) and 2023-01-15.
"""
import io
import re
import sys
import json
import time
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
RAW = EXT / "raw"
UA = {"User-Agent": "flight-fare-poc-research/0.1"}
REFRESH = "--refresh" in sys.argv

CITIES = {  # city: (IATA, state subdivision for `holidays`)
    "Delhi": ("DEL", "DL"), "Mumbai": ("BOM", "MH"), "Bangalore": ("BLR", "KA"), "Chennai": ("MAA", "TN"),
    "Hyderabad": ("HYD", "TG"), "Kolkata": ("CCU", "WB"), "Ahmedabad": ("AMD", "GJ")}
WIKIDATA = {"Delhi": "Q1353", "Mumbai": "Q1156", "Bangalore": "Q1355", "Chennai": "Q1352",
            "Hyderabad": "Q1361", "Kolkata": "Q1348", "Ahmedabad": "Q1070"}

SOURCES = {
    "holidays": ("python-holidays package (India, state subdivisions)", "https://github.com/vacanza/holidays",
                 "MIT. Dates incl. lunar/Islamic holidays are computed estimates and can be a day off the official gazette."),
    "approx": ("Hand-set approximate calendar rules (school board-exam / vacation / wedding season)", "n/a (rule of thumb)",
               "Approximate; school calendars differ by state and board, wedding muhurats differ by year. NOT a real dataset."),
    "cricsheet": ("Cricsheet ball-by-ball match data (IPL, Tests, ODIs, T20Is)", "https://cricsheet.org/downloads/",
                  "Open Data Commons Attribution License (https://cricsheet.org/licensing/). Attribution required."),
    "openmeteo": ("Open-Meteo Historical Weather API (ERA5 reanalysis)", "https://open-meteo.com/en/docs/historical-weather-api",
                  "CC BY 4.0 data, but the free API is for NON-COMMERCIAL use only (https://open-meteo.com/en/terms). Do not use commercially without a paid plan."),
    "fred": ("FRED / US EIA & Federal Reserve series: DJFUELUSGULF (US Gulf Coast jet fuel $/gal), DCOILBRENTEU (Brent $/bbl), DEXINUS (INR per USD)",
             "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF",
             "Public data from EIA and the Federal Reserve Board; FRED terms allow use with source credit. No API key used (public CSV endpoint). Jet fuel is a US-Gulf price, a proxy for Indian ATF (not the Indian ATF price itself)."),
    "ourairports": ("OurAirports airports.csv / runways.csv", "https://ourairports.com/data/", "Public domain (https://ourairports.com/data/)."),
    "openflights": ("OpenFlights routes.dat", "https://openflights.org/data.html",
                    "Open Database License (ODbL). The route snapshot is old (about 2014), so route counts are a rough hub-ness measure."),
    "wikidata": ("Wikidata P1082 (population, Census of India 2011 values)", "https://www.wikidata.org/wiki/Special:EntityData/Q1353.json",
                 "CC0. City-proper population, Census 2011."),
}


# --------------------------------------------------------------------------- helpers
def get(url, params=None, tries=6, headers=UA):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=120)
        except requests.RequestException:
            time.sleep(10 * (i + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up on {url}")


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not REFRESH:
        return path
    print("  downloading", url, flush=True)
    path.write_bytes(get(url).content)
    return path


def cached(name, fn):
    p = EXT / name
    if p.exists() and not REFRESH:
        return pd.read_parquet(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = fn()
    df.to_parquet(p)
    return df


# --------------------------------------------------------------------------- 1. calendar
FESTIVALS = {  # name -> regex on holiday names (package names), window days relative to the festival (before, after)
    "diwali": (r"diwali|deepavali", (5, 2)), "holi": (r"^holi", (4, 2)), "dussehra": (r"dussehra|vijaya ?dashami|navratri|dasara", (6, 1)),
    "durga_puja": (r"durga|maha (saptami|ashtami|navami)|vijayadashami", (5, 1)), "eid": (r"id-ul|eid|bakrid|bakr", (4, 2)),
    "christmas": (r"christmas", (5, 5)), "pongal_sankranti": (r"pongal|sankranti|lohri|bihu", (3, 2)),
    "onam": (r"onam", (4, 2)), "ganesh_chaturthi": (r"ganesh", (4, 2)), "maha_shivratri": (r"shivaratri|shivratri", (3, 1)),
    "raksha_bandhan": (r"raksha", (3, 1)), "independence_day": (r"independence day", (3, 1)), "republic_day": (r"republic day", (3, 1)),
    "good_friday": (r"good friday", (2, 2)), "ugadi": (r"ugadi|gudi padwa|telugu new year", (3, 1))}


def fixed_date_events(years):  # Valentine's Day is a fixed date, not a gazetted holiday
    return {pd.Timestamp(y, 2, 14): "valentines" for y in years}


def build_calendar(d0="2021-10-01", d1="2024-03-31"):
    import holidays
    dates = pd.date_range(d0, d1)
    years = range(dates.min().year, dates.max().year + 1)
    out = []
    # festival dates are taken from the union of all state calendars used + Kerala (Onam) + Maharashtra etc.
    fest_dates = {f: set() for f in FESTIVALS}
    for sub in list(dict.fromkeys([c[1] for c in CITIES.values()] + ["KL", "PB", "AS", None])):
        for d, name in holidays.India(subdiv=sub, years=years).items():
            for f, (rx, _) in FESTIVALS.items():
                if re.search(rx, name.lower()):
                    fest_dates[f].add(pd.Timestamp(d))
    fest_dates["valentines"] = set(fixed_date_events(years))
    FESTIVALS.setdefault("valentines", (r"", (3, 1)))
    fest = pd.DataFrame(index=dates)
    for f, ds in fest_dates.items():
        ds = sorted(ds)
        before, after = FESTIVALS[f][1]
        rel = pd.Series(np.nan, index=dates)
        for d in ds:  # signed days relative to the nearest festival day
            r = (dates - d).days.values.astype(float)
            m = np.abs(r) < np.abs(rel.fillna(1e9).values)
            rel[m] = r[m]
        fest[f"fest_{f}_window"] = ((rel >= -before) & (rel <= after)).astype(int)
    fest["fest_any_major_window"] = fest.filter(like="_window").drop(columns=["fest_valentines_window"]).max(axis=1)
    major = sorted(set().union(*[v for k, v in fest_dates.items() if k not in ("valentines", "republic_day", "independence_day", "good_friday")]))
    major = pd.DatetimeIndex(major)
    fest["fest_days_to_next_major"] = [min([(m - d).days for m in major if m >= d] or [np.nan]) for d in dates]
    fest["fest_days_since_major"] = [min([(d - m).days for m in major if m <= d] or [np.nan]) for d in dates]
    md = dates.month * 100 + dates.day
    fest["fest_school_exam_season"] = ((md >= 215) & (md <= 331)).astype(int)       # approx CBSE/ICSE board exams
    fest["fest_school_summer_vacation"] = ((md >= 415) & (md <= 630)).astype(int)   # approx
    fest["fest_school_winter_vacation"] = ((md >= 1224) | (md <= 110)).astype(int)  # approx
    fest["fest_wedding_season"] = ((md >= 1115) | (md <= 228) | ((md >= 415) & (md <= 615))).astype(int)  # approx
    fest.index.name = "date"
    fest = fest.reset_index()

    for city, (_, sub) in CITIES.items():
        h = holidays.India(subdiv=sub, years=years)
        hol = pd.Series(dates.isin(pd.DatetimeIndex(list(h.keys()))).astype(int), index=dates)
        nonwork = ((dates.dayofweek >= 5) | (hol.values == 1)).astype(int)
        nw = pd.Series(nonwork, index=dates)
        # maximal runs of non-working days
        run_id = (nw != nw.shift()).cumsum()
        run_len = nw.groupby(run_id).transform("size") * nw
        run_has_hol = hol.groupby(run_id).transform("max") * nw
        long_wknd = ((run_len >= 3) & (run_has_hol == 1)).astype(int)
        bridge = ((nw == 0) & (nw.shift(1) == 1) & (nw.shift(-1) == 1) & ((hol.shift(1) == 1) | (hol.shift(-1) == 1))).astype(int)
        # day before a long weekend starts / last day of a long weekend
        eve = ((long_wknd.shift(-1) == 1) & (long_wknd == 0)).astype(int)
        endd = ((long_wknd == 1) & (long_wknd.shift(-1) == 0)).astype(int)
        hd = np.array(sorted(pd.Timestamp(k) for k in h.keys()))
        d2n = [min([(x - d).days for x in hd if x >= d] or [np.nan]) for d in dates]
        d2l = [min([(d - x).days for x in hd if x <= d] or [np.nan]) for d in dates]
        c = pd.DataFrame({"city": city, "date": dates, "cal_is_holiday": hol.values,
                          "cal_days_to_next_holiday": d2n, "cal_days_since_holiday": d2l,
                          "cal_is_weekend": (dates.dayofweek >= 5).astype(int), "cal_long_weekend": long_wknd.values,
                          "cal_bridge_day": bridge.values, "cal_eve_of_long_weekend": eve.values, "cal_last_day_long_weekend": endd.values})
        for n in (3, 7, 14):
            c[f"cal_holidays_next{n}d"] = hol[::-1].rolling(n + 1, min_periods=1).sum()[::-1].values - hol.values
        out.append(c)
    cal = pd.concat(out, ignore_index=True)
    return cal.merge(fest, on="date")


# --------------------------------------------------------------------------- 1b. cricket
def build_cricket():
    rows = []
    for f in ["ipl", "tests", "odis", "t20s"]:
        p = download(f"https://cricsheet.org/downloads/{f}_json.zip", RAW / "cricsheet" / f"{f}_json.zip")
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            i = json.loads(z.read(n))["info"]
            d = i["dates"][0]
            if "2022-01-01" <= d <= "2023-12-31":
                teams = i.get("teams", [])
                ev = i.get("event", {}).get("name", "") if isinstance(i.get("event"), dict) else ""
                rows.append({"date": pd.Timestamp(d), "city": i.get("city"), "match_type": i.get("match_type"),
                             "gender": i.get("gender"), "ipl": int("Indian Premier League" in ev),
                             "india_home_intl": int("India" in teams and "Indian Premier League" not in ev)})
    m = pd.DataFrame(rows)
    m["city"] = m.city.replace({"Navi Mumbai": "Mumbai", "Bengaluru": "Bangalore"})
    # only matches in India count: India-hosted = city is one of the Indian metros or IPL/India-home match
    return m


def cricket_features(cm):
    dates = pd.date_range("2022-01-01", "2023-12-31")
    day = cm[cm.gender == "male"].groupby("date").agg(crk_ipl_day=("ipl", "max"), crk_india_playing_day=("india_home_intl", "max"))
    base = pd.DataFrame(index=dates).join(day).fillna(0).astype(int)
    out = []
    for city in CITIES:
        s = cm[(cm.city == city) & (cm.gender == "male")].groupby("date").size().reindex(dates).fillna(0).clip(upper=1)
        c = pd.DataFrame({"city": city, "date": dates, "crk_match_in_city": s.values.astype(int)})
        c["crk_match_in_city_pm1"] = s.rolling(3, center=True, min_periods=1).max().values.astype(int)
        out.append(c)
    f = pd.concat(out)
    return f.merge(base.reset_index(names="date"), on="date")


# --------------------------------------------------------------------------- 2. weather
def airport_coords():
    a = pd.read_csv(download("https://davidmegginson.github.io/ourairports-data/airports.csv", RAW / "geo/airports.csv"))
    iata = {v[0]: k for k, v in CITIES.items()}
    a = a[a.iata_code.isin(iata)].copy()
    a = a[a.type == "large_airport"]
    a["city"] = a.iata_code.map(iata)
    return a.set_index("city")


DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,weather_code"
HOURLY = "relative_humidity_2m,wind_speed_10m"


def fetch_weather_one(city, lat, lon):
    parts = []
    for y in range(2015, 2024):
        s, e = f"{y}-01-01", (f"{y}-12-31" if y < 2023 else "2023-03-31")
        j = get("https://archive-api.open-meteo.com/v1/archive",
                {"latitude": lat, "longitude": lon, "start_date": s, "end_date": e, "daily": DAILY, "hourly": HOURLY,
                 "timezone": "Asia/Kolkata"}).json()
        d = pd.DataFrame(j["daily"]).rename(columns={"time": "date"})
        d["date"] = pd.to_datetime(d.date)
        h = pd.DataFrame(j["hourly"]); h["time"] = pd.to_datetime(h.time)
        h["date"] = h.time.dt.normalize()
        early = h[h.time.dt.hour <= 9]  # fog forms overnight / early morning
        fog = early.assign(f=((early.relative_humidity_2m >= 95) & (early.wind_speed_10m < 8)).astype(int)).groupby("date").f.sum()
        d["fog_proxy_hours"] = d.date.map(fog)
        parts.append(d)
        time.sleep(2)
    d = pd.concat(parts, ignore_index=True)
    d["city"] = city
    print("  weather", city, len(d), "days", flush=True)
    return d


def build_weather():
    co = airport_coords()
    parts = []
    for c in CITIES:  # one cached file per airport so an interrupted run resumes
        parts.append(cached(f"weather/{c}.parquet", lambda c=c: fetch_weather_one(c, co.loc[c, "latitude_deg"], co.loc[c, "longitude_deg"])))
    return pd.concat(parts, ignore_index=True)


HEAVY_RAIN_MM, RAIN_DAY_MM, STORM_GUST_KMH = 15.6, 2.5, 60  # IMD "moderate or heavier" daily rain; rainy-day; gust threshold


def weather_flags(w):
    w = w.copy()
    w["rain_day"] = (w.precipitation_sum >= RAIN_DAY_MM).astype(int)
    w["heavy_rain"] = (w.precipitation_sum >= HEAVY_RAIN_MM).astype(int)
    w["storm"] = ((w.weather_code >= 95) | (w.wind_gusts_10m_max >= STORM_GUST_KMH)).astype(int)
    w["fog_day"] = (w.fog_proxy_hours >= 3).astype(int)
    return w


ACT_COLS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip_mm", "precipitation_hours": "rain_hours",
            "wind_speed_10m_max": "wind_max", "wind_gusts_10m_max": "gust_max", "weather_code": "wmo_code",
            "heavy_rain": "heavy_rain", "storm": "storm", "fog_proxy_hours": "fog_proxy_hours", "fog_day": "fog_day"}
CLIM_COLS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip_mm", "wind_gusts_10m_max": "gust_max",
             "rain_day": "rain_prob", "heavy_rain": "heavy_rain_prob", "storm": "storm_prob", "fog_day": "fog_prob"}


def weather_tables(w, target_dates):
    w = weather_flags(w)
    actual = w[w.date.between("2022-01-15", "2023-03-31")][["city", "date"] + list(ACT_COLS)].rename(columns=ACT_COLS)
    actual = actual.rename(columns={c: f"wxa_{c}" for c in actual.columns if c not in ("city", "date")})
    hist = w[w.date < "2022-01-01"].copy()  # 2015-2021 only: strictly before any journey date
    hist["doy_key"] = hist.date.dt.month * 100 + hist.date.dt.day
    rows = []
    for city, g in hist.groupby("city"):
        g = g.set_index("date")
        for t in target_dates:
            vals = []
            for y in range(2015, 2022):
                c = pd.Timestamp(y, t.month, min(t.day, 28)) + pd.Timedelta(days=max(t.day - 28, 0))
                vals.append(g.loc[c - pd.Timedelta(days=3): c + pd.Timedelta(days=3), list(CLIM_COLS)])
            m = pd.concat(vals).mean()
            rows.append({"city": city, "date": t, **{f"wxl_{CLIM_COLS[k]}": v for k, v in m.items()}})
    return actual, pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. fuel + fx
FRED = {"jet_usd_gal": "DJFUELUSGULF", "brent_usd_bbl": "DCOILBRENTEU", "usdinr": "DEXINUS"}


def build_fuel():
    cols = {}
    for k, sid in FRED.items():
        r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid, "cosd": "2021-06-01", "coed": "2023-04-30"}, headers=None)  # FRED drops custom User-Agents
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", k]
        d["date"] = pd.to_datetime(d.date)
        d[k] = pd.to_numeric(d[k], errors="coerce")
        cols[k] = d.set_index("date")[k]
        time.sleep(1)
    f = pd.DataFrame(cols).reindex(pd.date_range("2021-06-01", "2023-04-30")).ffill()  # carry last quote over weekends / holidays
    f["jet_inr_litre"] = f.jet_usd_gal * f.usdinr / 3.78541
    out = pd.DataFrame(index=f.index)
    for k in ["jet_usd_gal", "brent_usd_bbl", "usdinr", "jet_inr_litre"]:
        out[k] = f[k]
        out[f"{k}_chg7"] = f[k].pct_change(7)
        out[f"{k}_chg30"] = f[k].pct_change(30)
    out.index.name = "date"
    return out.reset_index()


FUEL_KEEP = ["jet_usd_gal", "jet_usd_gal_chg7", "jet_usd_gal_chg30", "brent_usd_bbl", "brent_usd_bbl_chg30", "usdinr", "usdinr_chg30",
             "jet_inr_litre", "jet_inr_litre_chg30"]


# --------------------------------------------------------------------------- 4. static airport / city
def build_static():
    a = pd.read_csv(download("https://davidmegginson.github.io/ourairports-data/airports.csv", RAW / "geo/airports.csv"))
    rw = pd.read_csv(download("https://davidmegginson.github.io/ourairports-data/runways.csv", RAW / "geo/runways.csv"))
    rt = pd.read_csv(download("https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat", RAW / "geo/routes.dat"),
                     header=None, names=["airline", "aid", "src", "sid", "dst", "did", "codeshare", "stops", "equip"])
    wd = get("https://www.wikidata.org/w/api.php", {"action": "wbgetentities", "ids": "|".join(WIKIDATA.values()), "props": "claims",
                                                     "format": "json"}).json()["entities"]
    pop = {}
    for city, q in WIKIDATA.items():
        for c in wd[q]["claims"]["P1082"]:
            t = c.get("qualifiers", {}).get("P585", [{}])[0].get("datavalue", {}).get("value", {}).get("time", "")
            if t.startswith("+2011"):
                pop[city] = float(c["mainsnak"]["datavalue"]["value"]["amount"])
    rows = []
    for city, (iata, _) in CITIES.items():
        ap = a[(a.iata_code == iata) & (a.type == "large_airport")].iloc[0]
        r = rw[(rw.airport_ident == ap.ident) & (rw.closed == 0)]
        rows.append({"city": city, "apt_elevation_ft": ap.elevation_ft, "apt_runways": len(r), "apt_longest_runway_ft": r.length_ft.max(),
                     "apt_routes_out": int((rt.src == iata).sum()), "apt_dest_count": rt[rt.src == iata].dst.nunique(),
                     "apt_city_pop_2011_m": pop[city] / 1e6})
    s = pd.DataFrame(rows)
    s["apt_hub_flag"] = (s.apt_dest_count >= s.apt_dest_count.median()).astype(int)  # above-median destinations among the 7 metros
    return s


# --------------------------------------------------------------------------- joins
def _sided(df, tbl, side, key, on_city, suffix_cols):
    """Merge a (city,date) table on journey_date for the origin ('o') or destination ('d') city."""
    t = tbl.rename(columns={c: (f"{c.split('_', 1)[0]}_{side}_{c.split('_', 1)[1]}") for c in suffix_cols}).rename(columns={"city": on_city})
    return df.merge(t, on=[on_city, "date"], how="left")


def enrich():
    df = pd.read_parquet(ROOT / "data/merged.parquet")
    df["date"] = df.journey_date
    df["obs_date"] = df.journey_date - pd.to_timedelta(df.days_left, unit="D")
    print("scrape dates:", df.groupby("period").obs_date.agg(["min", "max"]).to_dict("index"))
    n0 = len(df)

    print("1. calendar / holidays", flush=True)
    cal = cached("calendar.parquet", build_calendar)
    cal_city = ["cal_is_holiday", "cal_days_to_next_holiday", "cal_days_since_holiday", "cal_is_weekend", "cal_long_weekend", "cal_bridge_day",
                "cal_eve_of_long_weekend", "cal_last_day_long_weekend", "cal_holidays_next3d", "cal_holidays_next7d", "cal_holidays_next14d"]
    cal_day = [c for c in cal.columns if c.startswith("fest_")]
    for side, col in (("o", "source"), ("d", "destination")):
        df = _sided(df, cal[["city", "date"] + cal_city], side, None, col, cal_city)
    df = df.merge(cal[cal.city == "Delhi"][["date"] + cal_day], on="date", how="left")

    print("1b. cricket", flush=True)
    crk = cached("cricket_features.parquet", lambda: cricket_features(build_cricket()))
    crk_city = ["crk_match_in_city", "crk_match_in_city_pm1"]
    for side, col in (("o", "source"), ("d", "destination")):
        df = _sided(df, crk[["city", "date"] + crk_city], side, None, col, crk_city)
    df = df.merge(crk[crk.city == "Delhi"][["date", "crk_ipl_day", "crk_india_playing_day"]], on="date", how="left")

    print("2. weather (Open-Meteo)", flush=True)
    raw_w = cached("weather_raw.parquet", build_weather)
    actual, clim = weather_tables(raw_w, sorted(df.date.unique()))
    clim.to_parquet(EXT / "weather_climatology.parquet"); actual.to_parquet(EXT / "weather_actual.parquet")
    for side, col in (("o", "source"), ("d", "destination")):
        ac = [c for c in actual.columns if c.startswith("wxa_")]; cl = [c for c in clim.columns if c.startswith("wxl_")]
        df = _sided(df, actual, side, None, col, ac)
        df = _sided(df, clim, side, None, col, cl)

    print("3. fuel / FX (FRED)", flush=True)
    fuel = cached("fuel_fx.parquet", build_fuel)[["date"] + FUEL_KEEP]
    fj = fuel.rename(columns={c: f"fx_{c}" for c in FUEL_KEEP})
    df = df.merge(fj, on="date", how="left")
    fo = fuel.rename(columns={"date": "obs_date", **{c: f"fx_{c}_obs" for c in FUEL_KEEP}})
    df = df.merge(fo, on="obs_date", how="left")

    print("4. static airport / city", flush=True)
    st = cached("airport_static.parquet", build_static)
    sc = [c for c in st.columns if c.startswith("apt_")]
    for side, col in (("o", "source"), ("d", "destination")):
        df = df.merge(st[["city"] + sc].rename(columns={"city": col, **{c: f"apt_{side}_{c[4:]}" for c in sc}}), on=col, how="left")

    assert len(df) == n0, "join changed the row count"
    df = df.drop(columns=["date"])
    df.to_parquet(ROOT / "data/enriched.parquet")
    print("wrote data/enriched.parquet", df.shape, flush=True)
    return df


# --------------------------------------------------------------------------- feature groups (used by train_v2.py)
def groups(cols):
    g = {
        "calendar": [c for c in cols if c.startswith(("cal_", "fest_", "crk_"))],
        "weather_lag": [c for c in cols if c.startswith("wxl_")],
        "weather_actual": [c for c in cols if c.startswith("wxa_")],
        "fuel_fx": [c for c in cols if c.startswith("fx_")],
        "airport_static": [c for c in cols if c.startswith("apt_")],
    }
    return g


def nonconstant(df, cols, sub=None):
    """Columns with more than one distinct value in the given frame (a constant column cannot help a tree)."""
    d = df if sub is None else df[sub]
    return [c for c in cols if d[c].nunique(dropna=False) > 1]


DESC = {  # base name (side prefix removed) -> (description, source key)
    "cal_is_holiday": ("1 if the journey date is a public holiday in the city's state (or national)", "holidays"),
    "cal_days_to_next_holiday": ("days until the next holiday for that city (0 = today)", "holidays"),
    "cal_days_since_holiday": ("days since the last holiday for that city", "holidays"),
    "cal_is_weekend": ("1 if Saturday or Sunday", "holidays"),
    "cal_long_weekend": ("1 if the date is inside a run of >=3 consecutive non-working days that includes a weekday holiday", "holidays"),
    "cal_bridge_day": ("1 if a working day sandwiched between non-working days next to a holiday", "holidays"),
    "cal_eve_of_long_weekend": ("1 on the working day before a long weekend starts (outbound rush)", "holidays"),
    "cal_last_day_long_weekend": ("1 on the last day of a long weekend (return rush)", "holidays"),
    "cal_holidays_next3d": ("number of holidays in the next 3 days (excluding today)", "holidays"),
    "cal_holidays_next7d": ("number of holidays in the next 7 days", "holidays"),
    "cal_holidays_next14d": ("number of holidays in the next 14 days", "holidays"),
    "crk_match_in_city": ("1 if an IPL / international men's cricket match starts in that city on the date (Cricsheet)", "cricsheet"),
    "crk_match_in_city_pm1": ("same, within +-1 day", "cricsheet"),
    "crk_ipl_day": ("1 if any IPL match is played anywhere that day (IPL 2022 started 26 Mar; IPL 2023 started 31 Mar, after the data)", "cricsheet"),
    "crk_india_playing_day": ("1 if India's men's team plays an international match that day (home or away)", "cricsheet"),
    "wxa_tmax": ("ACTUAL max temperature 2 m, deg C", "openmeteo"), "wxa_tmin": ("ACTUAL min temperature, deg C", "openmeteo"),
    "wxa_precip_mm": ("ACTUAL daily precipitation, mm", "openmeteo"), "wxa_rain_hours": ("ACTUAL hours with precipitation", "openmeteo"),
    "wxa_wind_max": ("ACTUAL max 10 m wind, km/h", "openmeteo"), "wxa_gust_max": ("ACTUAL max gust, km/h", "openmeteo"),
    "wxa_wmo_code": ("ACTUAL dominant WMO weather code", "openmeteo"),
    "wxa_heavy_rain": ("ACTUAL 1 if precipitation >= 15.6 mm (IMD 'moderate or heavier')", "openmeteo"),
    "wxa_storm": ("ACTUAL 1 if WMO code >= 95 (thunderstorm) or gust >= 60 km/h", "openmeteo"),
    "wxa_fog_proxy_hours": ("ACTUAL hours 00-09 local with RH >= 95% and wind < 8 km/h (fog proxy; ERA5 has no visibility)", "openmeteo"),
    "wxa_fog_day": ("ACTUAL 1 if fog proxy hours >= 3", "openmeteo"),
    "wxl_tmax": ("CLIMATOLOGY: mean max temp, same calendar day +-3 d, 2015-2021", "openmeteo"),
    "wxl_tmin": ("CLIMATOLOGY: mean min temp", "openmeteo"), "wxl_precip_mm": ("CLIMATOLOGY: mean daily precipitation, mm", "openmeteo"),
    "wxl_gust_max": ("CLIMATOLOGY: mean max gust, km/h", "openmeteo"),
    "wxl_rain_prob": ("CLIMATOLOGY: share of days with >= 2.5 mm rain", "openmeteo"),
    "wxl_heavy_rain_prob": ("CLIMATOLOGY: share of days with >= 15.6 mm", "openmeteo"),
    "wxl_storm_prob": ("CLIMATOLOGY: share of storm days", "openmeteo"), "wxl_fog_prob": ("CLIMATOLOGY: share of fog-proxy days", "openmeteo"),
    "apt_elevation_ft": ("airport elevation, ft", "ourairports"), "apt_runways": ("number of open runways", "ourairports"),
    "apt_longest_runway_ft": ("longest runway, ft", "ourairports"),
    "apt_routes_out": ("number of OpenFlights routes departing the airport (c.2014 snapshot)", "openflights"),
    "apt_dest_count": ("number of distinct destinations in OpenFlights routes", "openflights"),
    "apt_city_pop_2011_m": ("city-proper population, millions, Census 2011", "wikidata"),
    "apt_hub_flag": ("1 if destination count is at/above the median of the 7 metros", "openflights"),
}
FEST_DESC = "1 if the date is inside the festival window (days before/after the festival day), dates from the holidays package"


def describe(col):
    base = re.sub(r"^(cal|crk|wxa|wxl|apt)_[od]_", r"\1_", col)
    if base in DESC:
        return DESC[base]
    if col.startswith("fx_"):
        b = col[3:].removesuffix("_obs")
        when = "on the SCRAPE (observation) date" if col.endswith("_obs") else "on the JOURNEY date"
        return (f"{b} {when} (FRED); chgN = relative change vs N days earlier", "fred")
    if col.startswith("fest_"):
        if col.startswith(("fest_school", "fest_wedding")):
            return ("APPROXIMATE seasonal flag set by hand: " + col[5:].replace("_", " "), "approx")
        if col.endswith("_window") or col == "fest_any_major_window":
            return (FEST_DESC + ": " + col[5:], "holidays")
        return ("days to / since the nearest major festival (Diwali, Holi, Dussehra, Durga Puja, Eid, Christmas, Pongal, Onam, Ganesh, Shivratri, Raksha Bandhan, Ugadi)", "holidays")
    raise KeyError(col)


def write_docs(df):
    base_cols = set(pd.read_parquet(ROOT / "data/merged.parquet").columns)
    new = [c for c in df.columns if c not in base_cols and c != "obs_date"]
    lines = ["# Enriched features", "",
             f"Generated by `src/enrich.py` from `data/merged.parquet` ({len(df):,} rows). {len(new)} new columns. Cached source files live in `data/external/` (gitignored).",
             "", "`o_` = origin city, `d_` = destination city (inserted after the group prefix, e.g. `wxa_o_tmax`). Observation (scrape) date = journey date - days_left "
             "(2022-02-11 assumed, 2023-01-15), a single day per scrape, so `fx_*_obs` features are constants within a scrape. "
             "A feature that is constant in a training frame is dropped by `train_v2.py`.", "",
             "## Sources and licences", "", "| key | source | URL | licence / caveat |", "|---|---|---|---|"]
    for k, (n, u, l) in SOURCES.items():
        lines.append(f"| {k} | {n} | {u} | {l} |")
    lines += ["", "## Leakage notes", "",
              "- `wxa_*` (actual weather on the journey date) is NOT knowable at booking time. Treat results with it as an upper bound.",
              "- `wxl_*` uses only 2015-2021 data, so it is knowable at booking time.",
              "- `fx_*` (journey date) is the realised market price on the flight date; `fx_*_obs` is knowable at booking time.",
              "- `cal_*`, `fest_*`, `crk_*` are fixtures known in advance (cricket schedules are published ahead; here taken from played-match records).",
              "", "## Column dictionary", "", "| column | description | source | null rate | distinct values |", "|---|---|---|---|---|"]
    for c in new:
        d, s = describe(c)
        lines.append(f"| `{c}` | {d} | {s} | {df[c].isna().mean() * 100:.2f}% | {df[c].nunique()} |")
    (ROOT / "docs/ENRICHED_FEATURES.md").write_text("\n".join(lines) + "\n")
    print("wrote docs/ENRICHED_FEATURES.md", len(new), "columns")


if __name__ == "__main__":
    df = enrich()
    write_docs(df)
