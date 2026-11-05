"""Time-series features for the US Airfare Time Machine (price of the SAME flight on many search days).

    python src/ts_features.py external   # fetch/cache calendar, FRED fuel, Open-Meteo weather -> data/external_us/
    python src/ts_features.py base       # collapse quotes to one row per (flight, search day) in a DuckDB file
    python src/ts_features.py features   # history + route-market + external joins -> data/external_us/features.parquet
    python src/ts_features.py all

Entity "flight" = airline + route + departure date + departure time. Every feature is computable on the search
date: history features use only EARLIER search days of the same flight; route-market features use only EARLIER
search days; calendar/fuel use the search date; weather uses only a climatology (prior years) except the
clearly-labelled `wxa_*` ORACLE columns (actual weather on the departure date, NOT known at booking time).

Sources / licences (also in docs/TIMESERIES_REPORT.md):
  python-holidays (MIT) https://github.com/vacanza/holidays ; school-break windows are hand-set approximations.
  FRED public CSV (EIA / Fed data, cite source, no key) https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF
  Open-Meteo Historical API (ERA5; data CC BY 4.0; free API NON-COMMERCIAL only) https://open-meteo.com/en/terms
"""
import io
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/us-airfare-time-machine"
EXT = ROOT / "data/external_us"
DB = EXT / "ts.duckdb"
FEATS = EXT / "features.parquet"
EPOCH = pd.Timestamp("2026-01-01")
WX_VARS = {"temperature_2m_max": "tmax", "temperature_2m_min": "tmin", "precipitation_sum": "precip",
           "snowfall_sum": "snow", "wind_speed_10m_max": "wind"}
SOURCES = {
    "holidays": ("python-holidays (US federal) + hand-set approximate school-break windows", "https://github.com/vacanza/holidays",
                 "MIT. School-break windows (spring break, summer, Thanksgiving, winter) are rough national rules of thumb, not a dataset."),
    "fred": ("FRED / EIA: DJFUELUSGULF (US Gulf Coast kerosene-type jet fuel, $/gal), DCOILBRENTEU (Brent, $/bbl)",
             "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF",
             "Public EIA data via FRED's keyless CSV endpoint; credit the source. No key used."),
    "openmeteo": ("Open-Meteo Historical Weather API (ERA5 reanalysis), daily tmax/tmin/precip/snow/wind per airport",
                  "https://archive-api.open-meteo.com/v1/archive",
                  "Data CC BY 4.0, but the free API is for NON-COMMERCIAL use only (https://open-meteo.com/en/terms)."),
    "dataset": ("US Airfare Time Machine (Kaggle, 'other' licence: check the dataset page before commercial use)",
                "https://www.kaggle.com/datasets (us-airfare-time-machine)", "Licence listed as 'other' by Kaggle."),
}


def get(url, params=None, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=120)
        except requests.RequestException:
            time.sleep(8 * (i + 1)); continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (i + 1)); continue
        raise RuntimeError(f"{r.status_code} {url} {r.text[:200]}")
    raise RuntimeError(f"giving up on {url}")


# ------------------------------------------------------------------ calendar
def calendar_table(d0="2025-12-01", d1="2027-06-30"):
    import holidays
    dates = pd.date_range(d0, d1)
    hol = holidays.US(years=range(2025, 2028))
    hd = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in hol))
    # Easter (not federal) is a travel date; Black Friday is not a federal holiday either
    from dateutil.easter import easter
    easters = pd.DatetimeIndex([pd.Timestamp(easter(y)) for y in (2025, 2026, 2027)])
    is_h = dates.isin(hd)
    nonwork = (dates.dayofweek >= 5) | is_h
    # long weekend: a run of >=3 consecutive non-working days containing a weekday holiday
    run_id = (~nonwork).cumsum()
    run = pd.Series(nonwork, index=dates).groupby(run_id).transform("sum").values
    has_wd_h = pd.Series(is_h & (dates.dayofweek < 5), index=dates).groupby(run_id).transform("max").values
    c = pd.DataFrame(index=dates)
    c["is_holiday"] = is_h.astype(int)
    nxt = np.searchsorted(hd.values, dates.values, side="left")
    c["days_to_hol"] = np.where(nxt < len(hd), (hd.values[np.minimum(nxt, len(hd) - 1)] - dates.values).astype("timedelta64[D]").astype(float), np.nan)
    prv = np.searchsorted(hd.values, dates.values, side="right") - 1
    c["days_since_hol"] = np.where(prv >= 0, (dates.values - hd.values[np.maximum(prv, 0)]).astype("timedelta64[D]").astype(float), np.nan)
    c["hol_weekend"] = ((run >= 3) & has_wd_h.astype(bool) & nonwork).astype(int)
    c["days_to_easter"] = [min(abs((d - e).days) for e in easters) for d in dates]
    md = dates.month * 100 + dates.day
    c["school_summer"] = ((md >= 605) & (md <= 820)).astype(int)
    c["school_spring_break"] = ((md >= 307) & (md <= 412)).astype(int)
    c["school_winter"] = ((md >= 1220) | (md <= 103)).astype(int)
    tg = [pd.Timestamp(d) for d, n in hol.items() if "Thanksgiving" in n]
    c["thanksgiving_window"] = [int(any(-1 <= (d - t).days <= 4 for t in tg)) for d in dates]
    c["school_break_any"] = c[["school_summer", "school_spring_break", "school_winter", "thanksgiving_window"]].max(axis=1)
    c["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    c.index.name = "date"
    return c.reset_index()


# ------------------------------------------------------------------ fuel (FRED)
def fuel_table():
    out = {}
    for sid, k in [("DJFUELUSGULF", "jet"), ("DCOILBRENTEU", "brent")]:
        r = get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": sid, "cosd": "2025-10-01", "coed": "2026-11-01"})
        d = pd.read_csv(io.StringIO(r.text)); d.columns = ["date", k]
        d["date"] = pd.to_datetime(d.date); d[k] = pd.to_numeric(d[k], errors="coerce")
        out[k] = d.set_index("date")[k]
        time.sleep(1)
    f = pd.DataFrame(out).reindex(pd.date_range("2025-10-01", "2026-11-01")).ffill()  # carry the last quote over weekends/holidays
    for k in ("jet", "brent"):
        f[f"{k}_chg7"] = f[k].pct_change(7)
        f[f"{k}_chg30"] = f[k].pct_change(30)
    f.index.name = "date"
    return f.reset_index()


# ------------------------------------------------------------------ weather (Open-Meteo)
def weather_table(airports, y0=2016, last="2026-10-28"):
    parts = []
    for _, a in airports.iterrows():
        for y in range(y0, 2027):
            s, e = f"{y}-01-01", min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp(last)).strftime("%Y-%m-%d")
            j = get("https://archive-api.open-meteo.com/v1/archive",
                    {"latitude": a.latitude, "longitude": a.longitude, "start_date": s, "end_date": e,
                     "daily": ",".join(WX_VARS), "timezone": "auto"}).json()
            d = pd.DataFrame(j["daily"]).rename(columns={"time": "date", **WX_VARS})
            d["airport"] = a.iata
            parts.append(d)
            time.sleep(0.7)
        print("  weather", a.iata, flush=True)
    w = pd.concat(parts, ignore_index=True)
    w["date"] = pd.to_datetime(w.date)
    return w


def climatology(w, first_target_year=2026):
    """Same calendar day (+-3 days) averaged over years BEFORE the target year: knowable at booking time."""
    w = w[w.date.dt.year < first_target_year].copy()
    w["doy"] = w.date.dt.dayofyear.clip(upper=365)
    rows = []
    for ap, g in w.groupby("airport"):
        base = g.groupby("doy")[list(WX_VARS.values())].mean()
        base = base.reindex(range(1, 366)).interpolate(limit_direction="both")
        ext = pd.concat([base.iloc[-3:], base, base.iloc[:3]])
        sm = ext.rolling(7, center=True).mean().iloc[3:-3]
        sm["airport"] = ap
        rows.append(sm.reset_index())
    cl = pd.concat(rows, ignore_index=True)
    return cl.rename(columns={c: f"wxl_{c}" for c in WX_VARS.values()})


def build_external():
    EXT.mkdir(parents=True, exist_ok=True)
    ap = pd.read_csv(RAW / "airports.csv")
    if not (EXT / "calendar.parquet").exists():
        calendar_table().to_parquet(EXT / "calendar.parquet")
    if not (EXT / "fuel.parquet").exists():
        fuel_table().to_parquet(EXT / "fuel.parquet")
    if not (EXT / "weather_daily.parquet").exists():
        weather_table(ap).to_parquet(EXT / "weather_daily.parquet")
    w = pd.read_parquet(EXT / "weather_daily.parquet")
    climatology(w).to_parquet(EXT / "weather_clim.parquet")
    (EXT / "SOURCES.json").write_text(json.dumps(SOURCES, indent=1))
    print("external ready:", [p.name for p in EXT.glob("*.parquet")])


# ------------------------------------------------------------------ base table (DuckDB)
def con_open():
    import duckdb
    con = duckdb.connect(str(DB))
    con.execute("set threads=2; set memory_limit='7GB'; set temp_directory='%s'" % (EXT / "duck_tmp"))
    return con


def build_base():
    con = con_open()
    P = f"'{RAW / 'flight_prices.parquet'}'"
    con.execute(f"create or replace table air as select row_number() over (order by airline) - 1 as airline_id, airline from (select distinct airline from {P})")
    con.execute(f"create or replace table rt as select row_number() over (order by route) - 1 as route_id, route, origin, destination from (select distinct route, origin, destination from {P})")
    # one row per (flight, search day); ~1% of keys have >1 quote on a day (different fares/stops): keep the cheapest
    con.execute(f"""create or replace table q as
      select hash(p.route, p.airline, p.departure_date, p.departure_time) as fid,
             (p.search_date - DATE '2026-01-01')::SMALLINT as sdn,
             (p.departure_date - DATE '2026-01-01')::SMALLINT as ddn,
             rt.route_id::UTINYINT as route_id, air.airline_id::UTINYINT as airline_id,
             p.days_before_departure::SMALLINT as dbd,
             (split_part(p.departure_time, ':', 1)::INT * 60 + split_part(p.departure_time, ':', 2)::INT)::SMALLINT as dep_min,
             min(p.stops)::TINYINT as stops, min(p.duration_minutes)::SMALLINT as dur,
             min(p.price_usd)::SMALLINT as price
      from {P} p join rt using (route) join air using (airline)
      group by all""")
    print(con.sql("select count(*), count(distinct fid), min(sdn), max(sdn) from q").fetchall(), flush=True)
    con.close()


# ------------------------------------------------------------------ features (DuckDB + small pandas tables)
def route_market_tables(con):
    """Route / airline daily medians -> trailing 7 PREVIOUS search days (excludes today). Small tables, pandas."""
    rd = con.sql("select route_id, sdn, median(price) as med, count(*) as nq from q group by all").df()
    ad = con.sql("select route_id, airline_id, sdn, median(price) as amed from q group by all").df()
    days = np.arange(rd.sdn.min(), rd.sdn.max() + 1)
    med = rd.pivot(index="sdn", columns="route_id", values="med").reindex(days)
    prev7 = med.shift(1).rolling(7, min_periods=1).mean()  # previous 7 calendar days that have data (missing days skipped)
    r7 = prev7.stack().rename("route_med7").reset_index()
    cnt = rd.set_index(["sdn", "route_id"]).nq.unstack().reindex(days)
    day_nq = rd.groupby("sdn").nq.sum().reindex(days)
    cov = (day_nq / day_nq.rolling(15, center=True, min_periods=3).median()).rename("day_coverage").reset_index().rename(columns={"index": "sdn"})
    cov.columns = ["sdn", "day_coverage"]
    ad = ad.merge(rd[["route_id", "sdn", "med"]], on=["route_id", "sdn"])
    ad["rel"] = ad.amed / ad.med
    ad["rank_pct"] = ad.groupby(["route_id", "sdn"]).amed.rank(pct=True)
    out = []
    for (rid, aid), g in ad.groupby(["route_id", "airline_id"]):
        g = g.set_index("sdn").reindex(days)
        o = pd.DataFrame({"airline_rel7": g.rel.shift(1).rolling(7, min_periods=1).mean(),
                          "airline_rank7": g.rank_pct.shift(1).rolling(7, min_periods=1).mean()})
        o["route_id"], o["airline_id"] = rid, aid
        out.append(o.reset_index().rename(columns={"index": "sdn"}))
    arank = pd.concat(out, ignore_index=True)
    arank = arank.dropna(subset=["airline_rel7"])
    return r7, arank, cov


def build_features():
    con = con_open()
    cal = pd.read_parquet(EXT / "calendar.parquet")
    fuel = pd.read_parquet(EXT / "fuel.parquet")
    wact = pd.read_parquet(EXT / "weather_daily.parquet")
    wcl = pd.read_parquet(EXT / "weather_clim.parquet")
    for tag, df in (("s", cal), ("d", cal)):
        x = df.copy(); x.insert(0, "dn", (x.date - EPOCH).dt.days.astype("int16"))
        x = x.drop(columns=["date"]).rename(columns={c: f"{tag}_{c}" for c in x.columns if c != "dn"})
        con.register(f"cal_{tag}", x)
    f = fuel.copy(); f.insert(0, "dn", (f.date - EPOCH).dt.days.astype("int16")); f = f.drop(columns=["date"])
    f = f.rename(columns={c: f"fx_{c}" for c in f.columns if c != "dn"})
    con.register("fuel", f)
    a = wact.copy(); a.insert(0, "dn", (a.date - EPOCH).dt.days.astype("int32")); a = a.drop(columns=["date"])
    a = a[a.dn >= 0].rename(columns={v: f"wxa_{v}" for v in WX_VARS.values()})
    con.register("wact", a)
    wcl["doy"] = wcl.doy.astype("int16")
    con.register("wcl", wcl)
    r7, arank, cov = route_market_tables(con)
    con.register("r7", r7.rename(columns={"level_0": "sdn", "level_1": "route_id"}) if "route_id" not in r7.columns else r7)
    con.register("arank", arank)
    con.register("cov", cov)
    ap = pd.read_csv(RAW / "airports.csv"); rts = pd.read_csv(RAW / "routes.csv")
    con.register("rts_csv", rts[["route", "distance_miles"]])
    print("tables registered", flush=True)

    # ---- stage 1: per-flight history (window functions over earlier observations only)
    con.execute("""create or replace table hist as
      with c as (select *, case when price <> lag(price) over (partition by fid order by sdn) then 1 else 0 end as chg from q)
      select fid, sdn,
        lag(price) over w as last_price,
        sdn - lag(sdn) over w as last_gap,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 1 preceding) as p_1d,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 3 preceding) as p_3d,
        last_value(price) over (partition by fid order by sdn range between unbounded preceding and 7 preceding) as p_7d,
        min(price) over (partition by fid order by sdn rows between 3 preceding and 1 preceding) as min3,
        avg(price) over (partition by fid order by sdn rows between 3 preceding and 1 preceding) as mean3,
        stddev_samp(price) over (partition by fid order by sdn rows between 3 preceding and 1 preceding) as std3,
        min(price) over (partition by fid order by sdn rows between 7 preceding and 1 preceding) as min7,
        avg(price) over (partition by fid order by sdn rows between 7 preceding and 1 preceding) as mean7,
        stddev_samp(price) over (partition by fid order by sdn rows between 7 preceding and 1 preceding) as std7,
        min(price) over (partition by fid order by sdn rows between unbounded preceding and 1 preceding) as run_min,
        max(price) over (partition by fid order by sdn rows between unbounded preceding and 1 preceding) as run_max,
        max(case when chg = 1 then sdn end) over (partition by fid order by sdn rows between unbounded preceding and 1 preceding) as last_chg_sdn,
        sum(chg) over (partition by fid order by sdn rows between unbounded preceding and 1 preceding) as n_changes,
        first_value(sdn) over (partition by fid order by sdn) as first_sdn,
        row_number() over w - 1 as n_obs
      from c window w as (partition by fid order by sdn)""")
    print("hist done", flush=True)

    # ---- stage 2: join everything, write parquet
    con.execute("""create or replace table comp as
      select route_id, ddn, sdn, count(distinct airline_id) as n_airlines, count(*) as n_flights_day
      from q group by all""")
    sql = """
    copy (
     select q.fid, q.sdn, q.ddn, q.route_id, q.airline_id, q.price::FLOAT as price,
       q.dbd::FLOAT as dbd, q.stops::FLOAT as stops, q.dur::FLOAT as dur, (q.dep_min/60.0)::FLOAT as dep_hour,
       ((q.sdn + 3) % 7)::FLOAT as search_dow,           -- 2026-01-01 is a Thursday (dow 3 with Mon=0)
       ((q.ddn + 3) % 7)::FLOAT as dep_dow, month(DATE '2026-01-01' + q.ddn::INT)::FLOAT as dep_month,
       (rts_csv.distance_miles)::FLOAT as distance_mi,
       h.last_price::FLOAT as last_price, h.last_gap::FLOAT as last_gap, h.p_1d::FLOAT as p_1d, h.p_3d::FLOAT as p_3d, h.p_7d::FLOAT as p_7d,
       h.min3::FLOAT as min3, h.mean3::FLOAT as mean3, h.std3::FLOAT as std3, h.min7::FLOAT as min7, h.mean7::FLOAT as mean7, h.std7::FLOAT as std7,
       h.run_min::FLOAT as run_min, h.run_max::FLOAT as run_max,
       (q.sdn - coalesce(h.last_chg_sdn, h.first_sdn))::FLOAT as days_since_change,
       h.n_changes::FLOAT as n_changes, h.n_obs::FLOAT as n_obs,
       r7.route_med7::FLOAT as route_med7, arank.airline_rel7::FLOAT as airline_rel7, arank.airline_rank7::FLOAT as airline_rank7,
       comp.n_airlines::FLOAT as n_airlines, comp.n_flights_day::FLOAT as n_flights_day, cov.day_coverage::FLOAT as day_coverage,
       cs.* exclude (dn), cd.* exclude (dn),
       fx.* exclude (dn),
       wo.wxa_tmax as wxa_o_tmax, wo.wxa_tmin as wxa_o_tmin, wo.wxa_precip as wxa_o_precip, wo.wxa_snow as wxa_o_snow, wo.wxa_wind as wxa_o_wind,
       wd.wxa_tmax as wxa_d_tmax, wd.wxa_tmin as wxa_d_tmin, wd.wxa_precip as wxa_d_precip, wd.wxa_snow as wxa_d_snow, wd.wxa_wind as wxa_d_wind,
       co.wxl_tmax as wxl_o_tmax, co.wxl_tmin as wxl_o_tmin, co.wxl_precip as wxl_o_precip, co.wxl_snow as wxl_o_snow, co.wxl_wind as wxl_o_wind,
       cdd.wxl_tmax as wxl_d_tmax, cdd.wxl_tmin as wxl_d_tmin, cdd.wxl_precip as wxl_d_precip, cdd.wxl_snow as wxl_d_snow, cdd.wxl_wind as wxl_d_wind
     from q
     join hist h on h.fid = q.fid and h.sdn = q.sdn
     join rt on rt.route_id = q.route_id
     join rts_csv on rts_csv.route = rt.route
     left join r7 on r7.route_id = q.route_id and r7.sdn = q.sdn
     left join arank on arank.route_id = q.route_id and arank.airline_id = q.airline_id and arank.sdn = q.sdn
     left join comp on comp.route_id = q.route_id and comp.ddn = q.ddn and comp.sdn = q.sdn
     left join cov on cov.sdn = q.sdn
     left join cal_s cs on cs.dn = q.sdn
     left join cal_d cd on cd.dn = q.ddn
     left join fuel fx on fx.dn = q.sdn
     left join wact wo on wo.airport = rt.origin and wo.dn = q.ddn
     left join wact wd on wd.airport = rt.destination and wd.dn = q.ddn
     left join wcl co on co.airport = rt.origin and co.doy = least(dayofyear(DATE '2026-01-01' + q.ddn::INT), 365)
     left join wcl cdd on cdd.airport = rt.destination and cdd.doy = least(dayofyear(DATE '2026-01-01' + q.ddn::INT), 365)
    ) to '@@FEATS@@' (format parquet, compression zstd, row_group_size 500000)""".replace('@@FEATS@@', str(FEATS))
    con.execute(sql)
    n = con.sql(f"select count(*) from '{FEATS}'").fetchall()
    print("features written", n, FEATS, flush=True)
    con.close()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("external", "all"): build_external()
    if stage in ("base", "all"): build_base()
    if stage in ("features", "all"): build_features()
