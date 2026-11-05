# US service for the FareSight demo (`src/us_service.py`)

Backend only: the Flask app imports these functions and does the routing/UI. All data is real (US Airfare Time Machine, 2026); nothing synthetic.

## Read this first: what the demo will show

The model **barely improves on "the price stays the same"** (full numbers in `docs/TIMESERIES_REPORT.md`, sections 5 and 11). One day ahead it is no better than the listed price (93.2% vs 93.1% accuracy), so `predicted_price` is **almost always equal to `listed_price`** at the default horizon of 1 day (on 20,000 sampled snapshot flights, 97% had a predicted change under 1%). It starts to add value only 7 to 14 days ahead (+0.9 to +3.7 accuracy points over carrying the price forward). "May drop" flags use the lowest predicted price from the chosen horizon to day 14, so they come from the longer-horizon models: about 5.6% of snapshot flights are flagged at the default 10% threshold. In the September test window, flights the model expected to fall at least 10% in 14 days really fell in 62% of cases (35% for all flights) and fell at least 10% in 50% (22% for all), so a flag is informative but wrong about half the time. The UI should show "listed price" prominently, show the prediction as an estimate, and not claim savings as certain.

## API (`import us_service` with `src/` on the path)

| function | returns |
|---|---|
| `us_meta()` | supported origins/destinations, the 50 valid `pairs` ("LAX-SFO"), `airports`, `departure_date_range`, `listed_as_of`, `counts`, `feature_families` (title, column count, description), `feature_columns_total`, `horizon` (trained horizons and the per-horizon accuracy table vs last-price-carried-forward, with a plain-text `verdict` per horizon), `headline_metrics` (next-day model vs naive baselines), `defaults`, `honest_note`, `sources` (licences), `data_limits` |
| `us_search(origin, destination, date, cabin=None, config=None)` | `{ok, currency, listed_as_of, config, count, flights, note, error?}`. Unknown route or bad date: `ok: false`, `error`, empty `flights`. Date outside the snapshot's departures (2026-10-07 to 2027-03-29): `ok: true`, empty `flights`, explanatory `note`. |
| `us_curve(flight_id, config=None)` | one flight dict (same fields as in a search result), or `None` for an unknown id |

`config` keys (all optional): `predict_ahead_days` 1 to 14 (default 1), `flag_threshold_pct` (10), `big_save_threshold_pct` (20), `big_save_min_usd` (20; 0 switches the dollar test off). `cabin` is accepted and ignored: the dataset has no cabin field.

Flight fields: `id`, `airline`, `flight_code` (airline + departure time; there is no flight number), `source`, `destination`, `dep_date`, `dep_time`, `arr_time` (or null), `arr_day_offset`, `stops`, `duration_h`, `cabin` (null), `currency` ("USD"), `listed_price` (cheapest quote on the last search day), `listed_as_of`, `days_left_at_listing`, `predicted_price`, `predicted_horizon_days` (shorter than requested for flights departing within a few weeks of the listing date, because the data has no quotes closer than 7 days to departure), `change_pct`, `may_drop`, `drop_pct`, `drop_best_days_left` (days before departure at the cheapest predicted point), `drop_best_days_ahead`, `drop_price`, `big_save`, `save_usd`, `curve` (list of `{days_ahead, days_left, price}`, days 1 to 14, omitted where the target day would be closer than 7 days to departure), `inputs`.

Rules (same as the India side): `may_drop` when the lowest predicted price at or after the horizon (within the 14-day curve) is at least `flag_threshold_pct` below the listed price; `big_save` when `may_drop` and (`drop_pct` >= `big_save_threshold_pct` or `save_usd` >= `big_save_min_usd`).

`inputs` = the exact model features for that flight at the horizon used, grouped as `{basics, history, route_market, calendar, fuel, weather}`, each `{title, items: [{key, label, value}]}` (88 columns in total; percentages are in percent, `weather` values are climatological averages, not forecasts).

### Example (shortened; real output for LAX-SFO on 2026-11-15)

```json
{
 "ok": true,
 "currency": "USD",
 "listed_as_of": "2026-09-30",
 "config": {
  "predict_ahead_days": 1,
  "flag_threshold_pct": 10.0,
  "big_save_threshold_pct": 20.0,
  "big_save_min_usd": 20.0
 },
 "count": 44,
 "cabin": null,
 "note": null,
 "flights": [
  {
   "id": "ae69fcfd188a9090",
   "airline": "American Airlines",
   "flight_code": "American Airlines 14:04",
   "source": "LAX",
   "destination": "SFO",
   "dep_date": "2026-11-15",
   "dep_time": "14:04",
   "arr_time": "15:36",
   "arr_day_offset": 0,
   "stops": 0,
   "duration_h": 1.53,
   "cabin": null,
   "currency": "USD",
   "listed_price": 224.0,
   "listed_as_of": "2026-09-30",
   "days_left_at_listing": 46,
   "predicted_price": 224.0,
   "predicted_horizon_days": 1,
   "horizon_requested_days": 1,
   "change_pct": 0.0,
   "may_drop": true,
   "drop_pct": 12.3,
   "drop_best_days_left": 32,
   "drop_best_days_ahead": 14,
   "drop_price": 196.45,
   "big_save": true,
   "save_usd": 27.55,
   "curve": [
    {
     "days_ahead": 1,
     "days_left": 45,
     "price": 224.0
    },
    {
     "days_ahead": 2,
     "days_left": 44,
     "price": 223.85
    },
    {
     "days_ahead": 3,
     "days_left": 43,
     "price": 223.59
    },
    {
     "...": "14 points in total"
    }
   ],
   "inputs": {
    "basics": {
     "title": "Basics",
     "items": [
      {
       "key": "dbd",
       "label": "Days before departure (on the predicted day)",
       "value": 45
      },
      {
       "key": "search_dow",
       "label": "Weekday of the predicted day (0=Mon)",
       "value": 3
      },
      {
       "...": "11 items in total"
      }
     ]
    },
    "history": {
     "title": "Price history of this flight",
     "items": [
      {
       "key": "last_price",
       "label": "Last observed price",
       "value": 224.0
      },
      {
       "key": "p_1d",
       "label": "Price about 1 day earlier",
       "value": 219.0
      },
      {
       "...": "22 items in total"
      }
     ]
    },
    "route_market": {
     "title": "Route market",
     "items": [
      {
       "key": "lr_route",
       "label": "Log ratio last / route 7-day median",
       "value": 0.817
      },
      {
       "key": "route_med7",
       "label": "Route median price, last 7 search days",
       "value": 99.0
      },
      {
       "...": "7 items in total"
      }
     ]
    },
    "calendar": {
     "title": "Holidays and calendar",
     "items": [
      {
       "key": "s_is_holiday",
       "label": "Federal holiday (predicted day)",
       "value": 0
      },
      {
       "key": "s_days_to_hol",
       "label": "Days to next holiday (predicted day)",
       "value": 11
      },
      {
       "...": "22 items in total"
      }
     ]
    },
    "fuel": {
     "title": "Fuel prices",
     "items": [
      {
       "key": "fx_jet",
       "label": "Jet fuel, US Gulf Coast ($/gal)",
       "value": 4.677
      },
      {
       "key": "fx_brent",
       "label": "Brent crude ($/bbl)",
       "value": 115.91
      },
      {
       "...": "6 items in total"
      }
     ]
    },
    "weather": {
     "title": "Weather (climatology)",
     "items": [
      {
       "key": "wxl_o_tmax",
       "label": "Typical max temp (C), origin",
       "value": 21.447
      },
      {
       "key": "wxl_o_tmin",
       "label": "Typical min temp (C), origin",
       "value": 11.471
      },
      {
       "...": "10 items in total"
      }
     ]
    }
   }
  },
  "... 43 more"
 ]
}
```

## How it works

`data/external_us/snapshot.parquet` holds, for each of the 197,189 flights quoted on 2026-09-30 (all departures after that date), the features as of that day (history, route-market, calendar/fuel/weather are computed at request time from small tables stored in the model file). `models/fare_model_ts.joblib` holds one LightGBM model per horizon (1, 2, 3, 5, 7, 10, 14 days; others interpolated), stored as plain numpy tree arrays, plus the calendar, fuel and climatology tables. **LightGBM is not needed at serving time** (no libomp issue on macOS). Inference is about 30 ms for a typical search (14 horizons x ~20-45 flights).

Built by `src/us_horizon.py` (`base`, `train`, `snapshot`; features shared with serving via `us_service.make_features`). The served models are trained on 10% of flights (98,585), all target dates up to 2026-09-30; the evaluation models (train to 15 Aug) are what the metrics describe.

## Deployment

Copy these (paths relative to the repo root; override with env `US_SNAPSHOT` and `US_MODEL`):

| file | size |
|---|---|
| `src/us_service.py` | small |
| `models/fare_model_ts.joblib` | 3.3 MB |
| `data/external_us/snapshot.parquet` | 8.4 MB |
| `models/ts_metrics.json`, `models/ts_horizon_metrics.json` (optional, read by `us_meta()` for the headline and horizon numbers; without them those fields are null) | small |

Memory: about 275 MB peak RSS after loading and 20 searches. Load is lazy (first call about 0.3 to 0.4 s) and cached; thread-safe load.

Libraries (versions used): Python 3.12.3, numpy 2.5.4, pandas 3.1.0, pyarrow 26.0.1, joblib 1.6.0. Not needed: lightgbm, scikit-learn, duckdb.

Self-test: `python src/us_service_selftest.py` (meta, searches on six routes/dates, unknown route, dates before and after the data, config handling; prints timings; currently all pass, 13-30 ms per search).

## Caveats

US routes only (50), 2026 data, listed as of 2026-09-30 (so "today" in the demo is a fixed past snapshot). No seat availability or cabin. Fuel and route-market features are held at their as-of values across the horizon; weather is a climatology. Open-Meteo's free API data is for non-commercial use only. Single seed, 10% flight sample. Re-run `us_horizon.py` to refresh when the dataset is extended.
